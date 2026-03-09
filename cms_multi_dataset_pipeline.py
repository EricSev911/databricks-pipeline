# Databricks notebook source
# MAGIC %md
# MAGIC # CMS Inpatient + Outpatient Pipeline
# MAGIC Building a medallion pipeline (Bronze / Silver / Gold) with two public CMS Medicare datasets joined on provider CCN (CMS Certification Number). Focus area is Kaiser Permanente in California, with a San Diego market drill-down.
# MAGIC
# MAGIC **Source files** (both CY2023, released 2025):
# MAGIC - Medicare Inpatient Hospitals by Provider & Service (~145k rows) - Diagnosis-Related Groups (DRG) level charges, payments, discharges
# MAGIC - Medicare Outpatient Hospitals by Provider & Service (~115k rows) - Ambulatory Payment Classifications (APC) level charges, payments, services
# MAGIC
# MAGIC **Data note:** Both files have `Avg_Mdcr_Pymt_Amt` (what Medicare actually paid the provider), but only the inpatient file has `Avg_Tot_Pymt_Amt` (total from all payers). I'm sticking with Medicare payment for both settings so the comparisons are apples-to-apples. The reimbursement ratio throughout is Medicare payment / billed charge.
# MAGIC
# MAGIC **Note:** In production the notebook would just write the Delta tables. The `display()` / `SELECT` cells and charts wouldn't be here. Visualizations would live in Power BI connected to a SQL Warehouse. Including everything here to show the full output in a single run. On a personal note, I frequently use Python plotting libraries in Databricks to rapidly prototype views and check my work while I'm querying the data.
# MAGIC
# MAGIC ### Setup
# MAGIC 1. Run All - the first cell creates the schema/volume and downloads both CSVs directly from CMS

# COMMAND ----------

# MAGIC %md
# MAGIC ## Environment setup

# COMMAND ----------

import urllib.request, os

CATALOG = spark.sql("SELECT current_catalog()").collect()[0][0]
SCHEMA  = "cms_multi"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.raw_files")
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

VOL = f"/Volumes/{CATALOG}/{SCHEMA}/raw_files"

SOURCES = {
    "MUP_INP_RY25_P03_V10_DY23_PrvSvc.CSV":
        "https://data.cms.gov/sites/default/files/2025-05/ca1c9013-8c7c-4560-a4a1-28cf7e43ccc8/MUP_INP_RY25_P03_V10_DY23_PrvSvc.CSV",
    "MUP_OUT_RY25_P04_V10_DY23_Prov_Svc.csv":
        "https://data.cms.gov/sites/default/files/2025-08/bceaa5e1-e58c-4109-9f05-832fc5e6bbc8/MUP_OUT_RY25_P04_V10_DY23_Prov_Svc.csv",
}

for filename, url in SOURCES.items():
    dest = f"{VOL}/{filename}"
    if os.path.exists(dest):
        print(f"  already exists: {filename}")
    else:
        print(f"  downloading:    {filename} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"  saved:          {dest}")

print(f"\ncatalog : {CATALOG}")
print(f"schema  : {SCHEMA}")
print(f"volume  : {VOL}")

# COMMAND ----------

from pyspark.sql import functions as F
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "font.size":        10,
    "axes.titlesize":   12,
    "axes.labelsize":   10,
})

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Bronze - load raw CSVs into Delta
# MAGIC Using `inferSchema` and full overwrite since these are static yearly files. If this were a scheduled pipeline you'd want to define the schema explicitly so column changes in the source file don't break things silently.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Inpatient

# COMMAND ----------

inp_raw = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(f"{VOL}/MUP_INP*")
)

inp_bronze = (
    inp_raw
    .withColumn("_load_ts",  F.current_timestamp())
    .withColumn("_src_file", F.col("_metadata.file_path"))
)

inp_bronze.write.mode("overwrite").format("delta") \
    .option("overwriteSchema", True).saveAsTable("inpatient_bronze")

print(f"inpatient_bronze : {spark.table('inpatient_bronze').count():,} rows, {len(inp_bronze.columns)} cols")

# COMMAND ----------

display(spark.table("inpatient_bronze").limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Outpatient

# COMMAND ----------

out_raw = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(f"{VOL}/MUP_OUT*")
)

out_bronze = (
    out_raw
    .withColumn("_load_ts",  F.current_timestamp())
    .withColumn("_src_file", F.col("_metadata.file_path"))
)

out_bronze.write.mode("overwrite").format("delta") \
    .option("overwriteSchema", True).saveAsTable("outpatient_bronze")

print(f"outpatient_bronze: {spark.table('outpatient_bronze').count():,} rows, {len(out_bronze.columns)} cols")

# COMMAND ----------

display(spark.table("outpatient_bronze").limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Silver - clean, rename, enrich
# MAGIC Doing inpatient in SQL and outpatient with the PySpark DataFrame API so I get practice with both approaches.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Inpatient Silver (SQL)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE inpatient_silver AS
# MAGIC SELECT
# MAGIC     -- CCNs lose leading zeros when inferSchema reads them as integer
# MAGIC     LPAD(CAST(Rndrng_Prvdr_CCN AS STRING), 6, '0') AS provider_ccn,
# MAGIC     INITCAP(TRIM(Rndrng_Prvdr_Org_Name)) AS provider_name,
# MAGIC     INITCAP(TRIM(Rndrng_Prvdr_City)) AS city,
# MAGIC     UPPER(TRIM(Rndrng_Prvdr_State_Abrvtn)) AS state,
# MAGIC     Rndrng_Prvdr_Zip5 AS zip5,
# MAGIC     Rndrng_Prvdr_RUCA_Desc AS ruca_desc,
# MAGIC     CAST(DRG_Cd AS INT) AS drg_code,
# MAGIC     DRG_Desc AS drg_desc,
# MAGIC     CAST(Tot_Dschrgs AS INT) AS discharges,
# MAGIC     CAST(Avg_Submtd_Cvrd_Chrg AS DECIMAL(12,2)) AS avg_charge,
# MAGIC     CAST(Avg_Mdcr_Pymt_Amt AS DECIMAL(12,2)) AS avg_medicare_pymt,
# MAGIC     CASE WHEN Avg_Submtd_Cvrd_Chrg > 0
# MAGIC          THEN LEAST(ROUND(Avg_Mdcr_Pymt_Amt / Avg_Submtd_Cvrd_Chrg, 2), 1.0)
# MAGIC     END AS medicare_to_charge_ratio,
# MAGIC     (Avg_Mdcr_Pymt_Amt > Avg_Submtd_Cvrd_Chrg) AS pymt_exceeds_charge,
# MAGIC     ROUND(Avg_Submtd_Cvrd_Chrg * Tot_Dschrgs, 2) AS total_charges,
# MAGIC     ROUND(Avg_Mdcr_Pymt_Amt * Tot_Dschrgs, 2) AS total_medicare_pymt,
# MAGIC     _load_ts,
# MAGIC     _src_file
# MAGIC FROM inpatient_bronze
# MAGIC WHERE Rndrng_Prvdr_Org_Name IS NOT NULL
# MAGIC   AND DRG_Cd IS NOT NULL
# MAGIC   AND Tot_Dschrgs > 0;

# COMMAND ----------

cnt = spark.table("inpatient_silver").count()
flagged = spark.table("inpatient_silver").filter("pymt_exceeds_charge").count()
print(f"inpatient_silver : {cnt:,} rows")
print(f"  providers: {spark.table('inpatient_silver').select('provider_ccn').distinct().count():,}")
print(f"  states   : {spark.table('inpatient_silver').select('state').distinct().count()}")
print(f"  pymt > charge: {flagged:,} rows flagged (ratio capped at 1.0)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Outpatient Silver (PySpark DataFrame API)
# MAGIC Same cleaning logic as the SQL above. The charge column has a slightly different name in the outpatient file (`Avg_Tot_Sbmtd_Chrgs` vs `Avg_Submtd_Cvrd_Chrg`).

# COMMAND ----------

brz_out = spark.table("outpatient_bronze")

# standardize column names first
df_renamed = (
    brz_out
    .withColumnRenamed("Rndrng_Prvdr_CCN", "provider_ccn")
    .withColumnRenamed("Rndrng_Prvdr_Org_Name", "provider_name")
    .withColumnRenamed("Rndrng_Prvdr_City", "city")
    .withColumnRenamed("Rndrng_Prvdr_State_Abrvtn", "state")
    .withColumnRenamed("Rndrng_Prvdr_Zip5", "zip5")
    .withColumnRenamed("Rndrng_Prvdr_RUCA_Desc", "ruca_desc")
    .withColumnRenamed("APC_Cd", "apc_code")
    .withColumnRenamed("APC_Desc", "apc_desc")
    .withColumnRenamed("Bene_Cnt", "beneficiaries")
    .withColumnRenamed("CAPC_Srvcs", "capc_services")
    .withColumnRenamed("Avg_Tot_Sbmtd_Chrgs", "avg_charge")
    .withColumnRenamed("Avg_Mdcr_Pymt_Amt", "avg_medicare_pymt")
)

# cast data types and clean strings
df_cleaned = (
    df_renamed
    # CCNs lose leading zeros when inferSchema reads them as int
    .withColumn("provider_ccn", F.lpad(F.col("provider_ccn").cast("string"), 6, "0"))
    .withColumn("apc_code",       F.trim("apc_code"))
    .withColumn("beneficiaries",  F.col("beneficiaries").cast("int"))
    .withColumn("capc_services",  F.col("capc_services").cast("int"))
    .withColumn("avg_charge",        F.col("avg_charge").cast("decimal(12,2)"))
    .withColumn("avg_medicare_pymt", F.col("avg_medicare_pymt").cast("decimal(12,2)"))
    .withColumn("provider_name", F.initcap(F.trim("provider_name")))
    .withColumn("city",  F.initcap(F.trim("city")))
    .withColumn("state", F.upper(F.trim("state")))
)

# Falling back to F.expr for the ratio calculation because the nested PySpark syntax was getting a bit messy
df_metrics = (
    df_cleaned
    .withColumn("pymt_exceeds_charge", F.col("avg_medicare_pymt") > F.col("avg_charge"))
    .withColumn("medicare_to_charge_ratio", F.expr(
        "CASE WHEN avg_charge > 0 THEN LEAST(ROUND(avg_medicare_pymt / avg_charge, 2), 1.0) END"
    ))
    .withColumn("total_charges", F.round(F.col("avg_charge") * F.col("capc_services"), 2))
    .withColumn("total_medicare_pymt", F.round(F.col("avg_medicare_pymt") * F.col("capc_services"), 2))
)

# display(df_metrics.filter(F.col("medicare_to_charge_ratio").isNull())) # sanity check on the math
# print(f"Row count before dropping nulls: {df_metrics.count()}")

# final filter and select
outpatient_silver = (
    df_metrics
    .filter("provider_name IS NOT NULL AND apc_code IS NOT NULL AND capc_services > 0")
    .select(
        "provider_ccn", "provider_name", "city", "state", "zip5", "ruca_desc",
        "apc_code", "apc_desc", "beneficiaries", "capc_services",
        "avg_charge", "avg_medicare_pymt",
        "medicare_to_charge_ratio", "pymt_exceeds_charge",
        "total_charges", "total_medicare_pymt",
        "_load_ts", "_src_file"
    )
)

outpatient_silver.write.mode("overwrite").format("delta") \
    .option("overwriteSchema", True).saveAsTable("outpatient_silver")

cnt = spark.table("outpatient_silver").count()
flagged = spark.table("outpatient_silver").filter("pymt_exceeds_charge").count()
print(f"outpatient_silver: {cnt:,} rows")
print(f"  providers: {spark.table('outpatient_silver').select('provider_ccn').distinct().count():,}")
print(f"  pymt > charge: {flagged:,} rows flagged (ratio capped at 1.0)")


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Data quality checks
# MAGIC
# MAGIC **Note on scaling:** In reviewing Databricks optimization strategies for handling downstream data quality checks at massive scale (e.g., billions of rows), the UNION ALL pattern below becomes an I/O bottleneck. A more resourceful production approach would be refactoring into a single-pass aggregation to avoid redundant table scans, or leveraging Delta Live Tables (DLT) Expectations to handle constraints natively. In production these would also write to a monitoring table and trigger alerts rather than just printing results, so failures don't go unnoticed.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- basic checks on key columns. if any of these fail, the Silver filters missed something and the Gold numbers can't be trusted
# MAGIC CREATE OR REPLACE TABLE dq_results AS
# MAGIC SELECT *,
# MAGIC     ROUND(100.0 * violations / total, 3) AS pct,
# MAGIC     CASE WHEN violations = 0 THEN 'PASS' ELSE 'FAIL' END AS status
# MAGIC FROM (
# MAGIC     SELECT 'inpatient_silver' AS tbl, 'null provider_ccn' AS rule,
# MAGIC            COUNT(*) AS total,
# MAGIC            SUM(CASE WHEN provider_ccn IS NULL THEN 1 ELSE 0 END) AS violations
# MAGIC     FROM inpatient_silver
# MAGIC     UNION ALL
# MAGIC     SELECT 'inpatient_silver', 'drg_code out of range',
# MAGIC            COUNT(*),
# MAGIC            SUM(CASE WHEN drg_code NOT BETWEEN 1 AND 999 THEN 1 ELSE 0 END)
# MAGIC     FROM inpatient_silver
# MAGIC     UNION ALL
# MAGIC     SELECT 'inpatient_silver', 'avg_charge <= 0',
# MAGIC            COUNT(*),
# MAGIC            SUM(CASE WHEN avg_charge <= 0 THEN 1 ELSE 0 END)
# MAGIC     FROM inpatient_silver
# MAGIC     UNION ALL
# MAGIC     SELECT 'outpatient_silver', 'null provider_ccn',
# MAGIC            COUNT(*),
# MAGIC            SUM(CASE WHEN provider_ccn IS NULL THEN 1 ELSE 0 END)
# MAGIC     FROM outpatient_silver
# MAGIC     UNION ALL
# MAGIC     SELECT 'outpatient_silver', 'null apc_code',
# MAGIC            COUNT(*),
# MAGIC            SUM(CASE WHEN apc_code IS NULL THEN 1 ELSE 0 END)
# MAGIC     FROM outpatient_silver
# MAGIC     UNION ALL
# MAGIC     SELECT 'outpatient_silver', 'avg_charge <= 0',
# MAGIC            COUNT(*),
# MAGIC            SUM(CASE WHEN avg_charge <= 0 THEN 1 ELSE 0 END)
# MAGIC     FROM outpatient_silver
# MAGIC ) checks;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM dq_results ORDER BY tbl, rule;

# COMMAND ----------

# check how well the two datasets join, if match rate is low, the Gold layer joins will silently drop a lot of providers. In this case, I find the 88% match rate interesting. There's probably legitimate reasons as to why this is happening, but it showcases a good example of investigating if there are legitimate reasons for this mismatch.
inp_ccns = spark.table("inpatient_silver").select("provider_ccn").distinct()
out_ccns = spark.table("outpatient_silver").select("provider_ccn").distinct()

matched  = inp_ccns.intersect(out_ccns).count()
inp_only = inp_ccns.subtract(out_ccns).count()
out_only = out_ccns.subtract(inp_ccns).count()
total    = inp_ccns.union(out_ccns).distinct().count()

print("Cross-dataset join coverage (provider CCN)")
print(f"  Matched in both  : {matched:,}")
print(f"  Inpatient only   : {inp_only:,}")
print(f"  Outpatient only  : {out_only:,}")
print(f"  Match rate       : {100 * matched / total:.1f}%")

# COMMAND ----------

# how many rows did we lose going from Bronze to Silver? if this % is too high, something in the cleaning logic might be too aggressive. 
# I actually looked into the ~46% row drop. According to page 5, paragraph 1 of the dataset's methodology (https://data.cms.gov/sites/default/files/2024-06/54319dec-5705-40ce-b9a6-fd88043ef8ac/MUP_OUT_RY24_20240531_Methodology-508.pdf), CMS suppresses all numeric fields (charges, payments, service counts) when a provider-APC combination has fewer than 11 beneficiaries to protect patient privacy. These rows come through with blanks across every numeric column, so filtering on capc_services > 0 removes them. This accounts for the ~46% row drop between Bronze and Silver, which are intentionally empty, not data loss.
brz_inp_ct = spark.table("inpatient_bronze").count()
slv_inp_ct = spark.table("inpatient_silver").count()
brz_out_ct = spark.table("outpatient_bronze").count()
slv_out_ct = spark.table("outpatient_silver").count()

print("Bronze -> Silver row reconciliation")
print(f"  Inpatient : {brz_inp_ct:,} -> {slv_inp_ct:,}  ({brz_inp_ct - slv_inp_ct:,} dropped, {100*(brz_inp_ct - slv_inp_ct)/brz_inp_ct:.2f}%)")
print(f"  Outpatient: {brz_out_ct:,} -> {slv_out_ct:,}  ({brz_out_ct - slv_out_ct:,} dropped, {100*(brz_out_ct - slv_out_ct)/brz_out_ct:.2f}%)")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- make sure the natural keys are actually unique after cleaning. if there are duplicates, the Gold aggregates will double-count charges
# MAGIC SELECT 'inpatient' AS dataset,
# MAGIC        'provider_ccn + drg_code' AS natural_key,
# MAGIC        COUNT(*) AS total_rows,
# MAGIC        COUNT(DISTINCT CONCAT(provider_ccn, '-', drg_code)) AS distinct_keys,
# MAGIC        COUNT(*) - COUNT(DISTINCT CONCAT(provider_ccn, '-', drg_code)) AS duplicates
# MAGIC FROM inpatient_silver
# MAGIC UNION ALL
# MAGIC SELECT 'outpatient',
# MAGIC        'provider_ccn + apc_code',
# MAGIC        COUNT(*),
# MAGIC        COUNT(DISTINCT CONCAT(provider_ccn, '-', apc_code)),
# MAGIC        COUNT(*) - COUNT(DISTINCT CONCAT(provider_ccn, '-', apc_code))
# MAGIC FROM outpatient_silver;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- sanity check on charge distributions before building Gold aggregates. if the max is wildly higher than p75, worth investigating before trusting the averages, but considering medical treatments can vary wildly in costs, It's probably normal to have outliers.
# MAGIC SELECT 'inpatient' AS dataset,
# MAGIC        COUNT(*) AS rows,
# MAGIC        ROUND(MIN(avg_charge), 2) AS min_charge,
# MAGIC        ROUND(PERCENTILE_APPROX(avg_charge, 0.25), 2) AS p25,
# MAGIC        ROUND(PERCENTILE_APPROX(avg_charge, 0.50), 2) AS median_charge,
# MAGIC        ROUND(PERCENTILE_APPROX(avg_charge, 0.75), 2) AS p75,
# MAGIC        ROUND(MAX(avg_charge), 2) AS max_charge
# MAGIC FROM inpatient_silver
# MAGIC UNION ALL
# MAGIC SELECT 'outpatient',
# MAGIC        COUNT(*),
# MAGIC        ROUND(MIN(avg_charge), 2),
# MAGIC        ROUND(PERCENTILE_APPROX(avg_charge, 0.25), 2),
# MAGIC        ROUND(PERCENTILE_APPROX(avg_charge, 0.50), 2),
# MAGIC        ROUND(PERCENTILE_APPROX(avg_charge, 0.75), 2),
# MAGIC        ROUND(MAX(avg_charge), 2)
# MAGIC FROM outpatient_silver;

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Gold - Kaiser Permanente focus
# MAGIC Joining inpatient and outpatient on provider CCN, then narrowing to Kaiser facilities in California.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Kaiser CA overview
# MAGIC All KP facilities in California. Payment figures are Medicare payments (`Avg_Mdcr_Pymt_Amt`) since that's the field both files share.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE gold_kaiser_ca_overview AS
# MAGIC WITH inp AS (
# MAGIC     SELECT provider_ccn, provider_name, city,
# MAGIC            SUM(discharges) AS inp_discharges,
# MAGIC            SUM(total_charges) AS inp_charges,
# MAGIC            SUM(total_medicare_pymt) AS inp_medicare_pymt,
# MAGIC            COUNT(DISTINCT drg_code) AS drg_count
# MAGIC     FROM inpatient_silver
# MAGIC     WHERE state = 'CA' AND LOWER(provider_name) LIKE '%kaiser%'
# MAGIC     GROUP BY provider_ccn, provider_name, city
# MAGIC ),
# MAGIC outp AS (
# MAGIC     SELECT provider_ccn,
# MAGIC            SUM(capc_services) AS out_services,
# MAGIC            SUM(total_charges) AS out_charges,
# MAGIC            SUM(total_medicare_pymt) AS out_medicare_pymt,
# MAGIC            COUNT(DISTINCT apc_code) AS apc_count
# MAGIC     FROM outpatient_silver
# MAGIC     WHERE state = 'CA' AND LOWER(provider_name) LIKE '%kaiser%'
# MAGIC     GROUP BY provider_ccn
# MAGIC )
# MAGIC SELECT
# MAGIC     i.provider_ccn,
# MAGIC     i.provider_name,
# MAGIC     i.city,
# MAGIC     i.inp_discharges,
# MAGIC     i.drg_count,
# MAGIC     ROUND(i.inp_charges, 0) AS inp_charges,
# MAGIC     ROUND(i.inp_medicare_pymt, 0) AS inp_medicare_pymt,
# MAGIC     o.out_services,
# MAGIC     o.apc_count,
# MAGIC     ROUND(o.out_charges, 0) AS out_charges,
# MAGIC     ROUND(o.out_medicare_pymt, 0) AS out_medicare_pymt,
# MAGIC     ROUND(i.inp_charges + COALESCE(o.out_charges, 0), 0) AS combined_charges,
# MAGIC     ROUND(i.inp_medicare_pymt + COALESCE(o.out_medicare_pymt, 0), 0) AS combined_medicare_pymt,
# MAGIC     ROUND(100.0 * i.inp_charges
# MAGIC           / NULLIF(i.inp_charges + COALESCE(o.out_charges, 0), 0), 1) AS pct_inpatient
# MAGIC FROM inp i
# MAGIC LEFT JOIN outp o ON i.provider_ccn = o.provider_ccn
# MAGIC ORDER BY combined_charges DESC;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM gold_kaiser_ca_overview;

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE gold_kaiser_ca_summary AS
# MAGIC SELECT
# MAGIC     COUNT(*) AS n_facilities,
# MAGIC     SUM(combined_charges) AS total_combined_charges,
# MAGIC     SUM(combined_medicare_pymt) AS total_medicare_pymt,
# MAGIC     SUM(inp_discharges) AS total_discharges,
# MAGIC     SUM(out_services) AS total_services,
# MAGIC     ROUND(SUM(combined_medicare_pymt) / NULLIF(SUM(combined_charges), 0), 4) AS reimb_rate
# MAGIC FROM gold_kaiser_ca_overview;

# COMMAND ----------

# Kaiser CA Overview Dashboard - metrics pre-computed in gold_kaiser_ca_summary
kpi = spark.table("gold_kaiser_ca_summary").toPandas()

if not kpi.empty:
    row = kpi.iloc[0]

    fig, axes = plt.subplots(2, 3, figsize=(16, 7))
    fig.suptitle("Kaiser Permanente - California Overview", fontsize=15, fontweight="bold")

    def kpi_panel(ax, value, label, fmt="$", color="#0072CE"):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis("off")
        if fmt == "$":
            txt = f"${value:,.0f}" if value < 1e6 else f"${value/1e6:,.1f}M"
        elif fmt == "#":
            txt = f"{int(value):,}"
        elif fmt == "%":
            txt = f"{value:.1%}"
        else:
            txt = str(value)
        ax.text(0.5, 0.55, txt, ha="center", va="center",
                fontsize=26, fontweight="bold", color=color)
        ax.text(0.5, 0.22, label, ha="center", va="center",
                fontsize=10, color="#555555")
        ax.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9,
                     fill=False, ec="#E0E0E0", lw=1.5, transform=ax.transAxes))

    kpi_panel(axes[0][0], int(row["n_facilities"]), "KP Facilities in CA", fmt="#")
    kpi_panel(axes[0][1], float(row["total_combined_charges"]), "Total Combined Charges")
    kpi_panel(axes[0][2], float(row["total_medicare_pymt"]), "Total Medicare Payments")
    kpi_panel(axes[1][0], int(row["total_discharges"]), "Inpatient Discharges", fmt="#")
    kpi_panel(axes[1][1], int(row["total_services"]), "Outpatient Services", fmt="#")
    kpi_panel(axes[1][2], float(row["reimb_rate"]), "Medicare Reimb. Rate\n(Pymt / Charge)", fmt="%", color="#D9534F")

    plt.tight_layout()
    display(fig)
    plt.close(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Kaiser vs CA market
# MAGIC Weighted avg charge per discharge (inpatient) and per service (outpatient), Kaiser vs everyone else in the state. Reimbursement ratio = Medicare payment / charge for both settings (apples-to-apples).

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE gold_kp_vs_ca_market AS
# MAGIC WITH inp_grp AS (
# MAGIC     SELECT
# MAGIC         CASE WHEN LOWER(provider_name) LIKE '%kaiser%'
# MAGIC              THEN 'Kaiser Permanente' ELSE 'Other CA Hospitals' END AS system_group,
# MAGIC         SUM(discharges) AS inp_discharges,
# MAGIC         SUM(total_charges) AS inp_charges,
# MAGIC         SUM(total_medicare_pymt) AS inp_medicare_pymt
# MAGIC     FROM inpatient_silver WHERE state = 'CA'
# MAGIC     GROUP BY system_group
# MAGIC ),
# MAGIC out_grp AS (
# MAGIC     SELECT
# MAGIC         CASE WHEN LOWER(provider_name) LIKE '%kaiser%'
# MAGIC              THEN 'Kaiser Permanente' ELSE 'Other CA Hospitals' END AS system_group,
# MAGIC         SUM(capc_services) AS out_services,
# MAGIC         SUM(total_charges) AS out_charges,
# MAGIC         SUM(total_medicare_pymt) AS out_medicare_pymt
# MAGIC     FROM outpatient_silver WHERE state = 'CA'
# MAGIC     GROUP BY system_group
# MAGIC )
# MAGIC SELECT
# MAGIC     i.system_group,
# MAGIC     i.inp_discharges,
# MAGIC     ROUND(i.inp_charges / i.inp_discharges, 0) AS inp_avg_charge_per_dc,
# MAGIC     ROUND(i.inp_medicare_pymt / i.inp_discharges, 0) AS inp_avg_medicare_per_dc,
# MAGIC     ROUND(i.inp_medicare_pymt / i.inp_charges, 2) AS inp_medicare_to_charge,
# MAGIC     o.out_services,
# MAGIC     ROUND(o.out_charges / o.out_services, 0) AS out_avg_charge_per_svc,
# MAGIC     ROUND(o.out_medicare_pymt / o.out_services, 0) AS out_avg_medicare_per_svc,
# MAGIC     ROUND(o.out_medicare_pymt / o.out_charges, 2) AS out_medicare_to_charge
# MAGIC FROM inp_grp i
# MAGIC JOIN out_grp o ON i.system_group = o.system_group
# MAGIC ORDER BY i.system_group;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM gold_kp_vs_ca_market;

# COMMAND ----------

# Chart: KP vs Other CA - grouped bar with metric labels
mkt = spark.table("gold_kp_vs_ca_market").toPandas()
for c in mkt.columns:
    if c != "system_group":
        mkt[c] = mkt[c].astype(float)

if not mkt.empty:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    x = range(len(mkt))
    w = 0.35

    def add_bar_labels(ax, bars, fmt="$"):
        for bar in bars:
            val = bar.get_height()
            if fmt == "$":
                lbl = f"${val:,.0f}"
            else:
                lbl = f"{val:.2f}"
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    lbl, ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax = axes[0]
    b1 = ax.bar([i - w/2 for i in x], mkt["inp_avg_charge_per_dc"], w,
                label="Charge", color="#0072CE")
    b2 = ax.bar([i + w/2 for i in x], mkt["inp_avg_medicare_per_dc"], w,
                label="Medicare Pymt", color="#5CB85C")
    add_bar_labels(ax, b1); add_bar_labels(ax, b2)
    ax.set_xticks(list(x))
    ax.set_xticklabels(mkt["system_group"], fontsize=8)
    ax.set_title("Inpatient - Avg per Discharge")
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
    ax.legend(fontsize=7)

    ax2 = axes[1]
    b3 = ax2.bar([i - w/2 for i in x], mkt["out_avg_charge_per_svc"], w,
                 label="Charge", color="#0072CE")
    b4 = ax2.bar([i + w/2 for i in x], mkt["out_avg_medicare_per_svc"], w,
                 label="Medicare Pymt", color="#5CB85C")
    add_bar_labels(ax2, b3); add_bar_labels(ax2, b4)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(mkt["system_group"], fontsize=8)
    ax2.set_title("Outpatient - Avg per Service")
    ax2.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
    ax2.legend(fontsize=7)

    ax3 = axes[2]
    b5 = ax3.bar([i - w/2 for i in x], mkt["inp_medicare_to_charge"], w,
                 label="Inpatient", color="#4A90D9")
    b6 = ax3.bar([i + w/2 for i in x], mkt["out_medicare_to_charge"], w,
                 label="Outpatient", color="#FFB347")
    add_bar_labels(ax3, b5, fmt="r"); add_bar_labels(ax3, b6, fmt="r")
    ax3.set_xticks(list(x))
    ax3.set_xticklabels(mkt["system_group"], fontsize=8)
    ax3.set_title("Medicare Pymt / Charge Ratio")
    ax3.legend(fontsize=7)

    plt.suptitle("Kaiser vs Other CA Hospitals", fontsize=13, fontweight="bold")
    plt.tight_layout()
    display(fig)
    plt.close(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Kaiser top DRGs (inpatient)
# MAGIC What does Kaiser treat most, and how do their charges compare to the California statewide average for the same DRGs?

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE gold_kaiser_top_drgs AS
# MAGIC WITH kp_drg AS (
# MAGIC     SELECT drg_code, drg_desc,
# MAGIC            SUM(discharges) AS kp_discharges,
# MAGIC            ROUND(SUM(total_charges) / SUM(discharges), 0) AS kp_avg_charge
# MAGIC     FROM inpatient_silver
# MAGIC     WHERE state = 'CA' AND LOWER(provider_name) LIKE '%kaiser%'
# MAGIC     GROUP BY drg_code, drg_desc
# MAGIC ),
# MAGIC ca_drg AS (
# MAGIC     SELECT drg_code,
# MAGIC            SUM(discharges) AS ca_discharges,
# MAGIC            ROUND(SUM(total_charges) / SUM(discharges), 0) AS ca_avg_charge
# MAGIC     FROM inpatient_silver
# MAGIC     WHERE state = 'CA'
# MAGIC     GROUP BY drg_code
# MAGIC )
# MAGIC SELECT
# MAGIC     k.drg_code, k.drg_desc,
# MAGIC     k.kp_discharges, k.kp_avg_charge,
# MAGIC     c.ca_discharges, c.ca_avg_charge,
# MAGIC     ROUND(k.kp_avg_charge - c.ca_avg_charge, 0) AS charge_diff,
# MAGIC     ROUND(100.0 * (k.kp_avg_charge - c.ca_avg_charge)
# MAGIC           / NULLIF(c.ca_avg_charge, 0), 1) AS pct_diff
# MAGIC FROM kp_drg k
# MAGIC JOIN ca_drg c ON k.drg_code = c.drg_code
# MAGIC ORDER BY k.kp_discharges DESC;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT drg_code, drg_desc, kp_discharges, kp_avg_charge,
# MAGIC        ca_avg_charge, charge_diff, pct_diff
# MAGIC FROM gold_kaiser_top_drgs
# MAGIC ORDER BY kp_discharges DESC
# MAGIC LIMIT 15;

# COMMAND ----------

# Chart: KP top DRGs - ordered by Kaiser avg charge
drg = (
    spark.sql("""
        SELECT drg_desc, kp_avg_charge, ca_avg_charge
        FROM gold_kaiser_top_drgs
        ORDER BY kp_avg_charge DESC LIMIT 10
    """).toPandas()
)

if not drg.empty:
    drg["kp_avg_charge"] = drg["kp_avg_charge"].astype(float)
    drg["ca_avg_charge"] = drg["ca_avg_charge"].astype(float)
    drg = drg.sort_values("kp_avg_charge", ascending=True)
    drg["short"] = drg["drg_desc"].str[:40]
    fig, ax = plt.subplots(figsize=(10, 5))
    y = range(len(drg))
    h = 0.35
    b1 = ax.barh([i + h/2 for i in y], drg["ca_avg_charge"].values, h,
                 label="CA Statewide Avg", color="#B0B0B0")
    b2 = ax.barh([i - h/2 for i in y], drg["kp_avg_charge"].values, h,
                 label="Kaiser Avg", color="#0072CE")
    for bar in b1:
        ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                f" ${bar.get_width():,.0f}", va="center", fontsize=6, color="#666666")
    for bar, ca_val in zip(b2, drg["ca_avg_charge"].values):
        kp_val = bar.get_width()
        pct = (kp_val / ca_val * 100) if ca_val else 0
        ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                f" ${kp_val:,.0f} ({pct:.0f}% of CA)",
                va="center", fontsize=7, fontweight="bold")
    ax.set_yticks(list(y))
    ax.set_yticklabels(drg["short"].values, fontsize=8)
    ax.set_xlabel("Avg Charge per Discharge ($)")
    ax.set_title("Kaiser Top 10 DRGs by Avg Charge - KP vs CA Statewide")
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
    ax.legend()
    plt.tight_layout()
    display(fig)
    plt.close(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Kaiser top APCs (outpatient)
# MAGIC Same comparison but for outpatient services by APC code.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE gold_kaiser_top_apcs AS
# MAGIC WITH kp_apc AS (
# MAGIC     SELECT apc_code, apc_desc,
# MAGIC            SUM(capc_services) AS kp_services,
# MAGIC            ROUND(SUM(total_charges) / SUM(capc_services), 0) AS kp_avg_charge
# MAGIC     FROM outpatient_silver
# MAGIC     WHERE state = 'CA' AND LOWER(provider_name) LIKE '%kaiser%'
# MAGIC     GROUP BY apc_code, apc_desc
# MAGIC ),
# MAGIC ca_apc AS (
# MAGIC     SELECT apc_code,
# MAGIC            SUM(capc_services) AS ca_services,
# MAGIC            ROUND(SUM(total_charges) / SUM(capc_services), 0) AS ca_avg_charge
# MAGIC     FROM outpatient_silver
# MAGIC     WHERE state = 'CA'
# MAGIC     GROUP BY apc_code
# MAGIC )
# MAGIC SELECT
# MAGIC     k.apc_code, k.apc_desc,
# MAGIC     k.kp_services, k.kp_avg_charge,
# MAGIC     c.ca_services, c.ca_avg_charge,
# MAGIC     ROUND(k.kp_avg_charge - c.ca_avg_charge, 0) AS charge_diff,
# MAGIC     ROUND(100.0 * (k.kp_avg_charge - c.ca_avg_charge)
# MAGIC           / NULLIF(c.ca_avg_charge, 0), 1) AS pct_diff
# MAGIC FROM kp_apc k
# MAGIC JOIN ca_apc c ON k.apc_code = c.apc_code
# MAGIC ORDER BY k.kp_services DESC;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT apc_code, apc_desc, kp_services, kp_avg_charge,
# MAGIC        ca_avg_charge, charge_diff, pct_diff
# MAGIC FROM gold_kaiser_top_apcs
# MAGIC ORDER BY kp_services DESC
# MAGIC LIMIT 15;

# COMMAND ----------

# Chart: KP top APCs - ordered by Kaiser avg charge
apc = (
    spark.sql("""
        SELECT apc_desc, kp_avg_charge, ca_avg_charge
        FROM gold_kaiser_top_apcs
        ORDER BY kp_avg_charge DESC LIMIT 10
    """).toPandas()
)

if not apc.empty:
    apc["kp_avg_charge"] = apc["kp_avg_charge"].astype(float)
    apc["ca_avg_charge"] = apc["ca_avg_charge"].astype(float)
    apc = apc.sort_values("kp_avg_charge", ascending=True)
    apc["short"] = apc["apc_desc"].str[:40]
    fig, ax = plt.subplots(figsize=(10, 5))
    y = range(len(apc))
    h = 0.35
    b1 = ax.barh([i + h/2 for i in y], apc["ca_avg_charge"].values, h,
                 label="CA Statewide Avg", color="#B0B0B0")
    b2 = ax.barh([i - h/2 for i in y], apc["kp_avg_charge"].values, h,
                 label="Kaiser Avg", color="#0072CE")
    for bar in b1:
        ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                f" ${bar.get_width():,.0f}", va="center", fontsize=6, color="#666666")
    for bar, ca_val in zip(b2, apc["ca_avg_charge"].values):
        kp_val = bar.get_width()
        pct = (kp_val / ca_val * 100) if ca_val else 0
        ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                f" ${kp_val:,.0f} ({pct:.0f}% of CA)",
                va="center", fontsize=7, fontweight="bold")
    ax.set_yticks(list(y))
    ax.set_yticklabels(apc["short"].values, fontsize=8)
    ax.set_xlabel("Avg Charge per Service ($)")
    ax.set_title("Kaiser Top 10 APCs by Avg Charge - KP vs CA Statewide")
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
    ax.legend()
    plt.tight_layout()
    display(fig)
    plt.close(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ### San Diego spotlight
# MAGIC All San Diego hospitals, inpatient + outpatient combined.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE gold_sd_market AS
# MAGIC WITH inp AS (
# MAGIC     SELECT provider_ccn, provider_name, city,
# MAGIC            SUM(discharges) AS inp_discharges,
# MAGIC            SUM(total_charges) AS inp_charges,
# MAGIC            SUM(total_medicare_pymt) AS inp_medicare_pymt
# MAGIC     FROM inpatient_silver
# MAGIC     WHERE LOWER(city) LIKE '%san d%' AND state = 'CA'
# MAGIC     GROUP BY provider_ccn, provider_name, city
# MAGIC ),
# MAGIC outp AS (
# MAGIC     SELECT provider_ccn,
# MAGIC            SUM(capc_services) AS out_services,
# MAGIC            SUM(total_charges) AS out_charges,
# MAGIC            SUM(total_medicare_pymt) AS out_medicare_pymt
# MAGIC     FROM outpatient_silver
# MAGIC     WHERE LOWER(city) LIKE '%san d%' AND state = 'CA'
# MAGIC     GROUP BY provider_ccn
# MAGIC )
# MAGIC SELECT
# MAGIC     i.provider_name, i.city,
# MAGIC     i.inp_discharges,
# MAGIC     ROUND(i.inp_charges, 0) AS inp_charges,
# MAGIC     ROUND(i.inp_medicare_pymt, 0) AS inp_medicare_pymt,
# MAGIC     COALESCE(o.out_services, 0) AS out_services,
# MAGIC     ROUND(COALESCE(o.out_charges, 0), 0) AS out_charges,
# MAGIC     ROUND(COALESCE(o.out_medicare_pymt, 0), 0) AS out_medicare_pymt,
# MAGIC     ROUND(i.inp_charges + COALESCE(o.out_charges, 0), 0) AS combined_charges,
# MAGIC     ROUND(i.inp_medicare_pymt + COALESCE(o.out_medicare_pymt, 0), 0) AS combined_medicare_pymt,
# MAGIC     ROUND(100.0 * i.inp_charges
# MAGIC           / NULLIF(i.inp_charges + COALESCE(o.out_charges, 0), 0), 1) AS pct_inpatient,
# MAGIC     CASE WHEN LOWER(i.provider_name) LIKE '%kaiser%'
# MAGIC          THEN 'Y' ELSE 'N' END AS kaiser_flag
# MAGIC FROM inp i
# MAGIC LEFT JOIN outp o ON i.provider_ccn = o.provider_ccn
# MAGIC ORDER BY combined_charges DESC;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT provider_name, inp_discharges, inp_charges, inp_medicare_pymt,
# MAGIC        out_services, out_charges, out_medicare_pymt,
# MAGIC        combined_charges, combined_medicare_pymt, kaiser_flag
# MAGIC FROM gold_sd_market
# MAGIC ORDER BY combined_charges DESC;

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE gold_sd_rankings AS
# MAGIC SELECT
# MAGIC     provider_name,
# MAGIC     kaiser_flag,
# MAGIC     combined_charges,
# MAGIC     combined_medicare_pymt,
# MAGIC     inp_discharges,
# MAGIC     out_services,
# MAGIC     ROUND(combined_medicare_pymt / NULLIF(combined_charges, 0), 4) AS reimb_rate,
# MAGIC     RANK() OVER (ORDER BY combined_charges DESC) AS rank_charges,
# MAGIC     RANK() OVER (ORDER BY combined_medicare_pymt DESC) AS rank_pymt,
# MAGIC     RANK() OVER (ORDER BY inp_discharges DESC) AS rank_discharges,
# MAGIC     RANK() OVER (ORDER BY out_services DESC) AS rank_services,
# MAGIC     RANK() OVER (ORDER BY combined_medicare_pymt / NULLIF(combined_charges, 0) DESC) AS rank_reimb_rate,
# MAGIC     COUNT(*) OVER () AS total_facilities
# MAGIC FROM gold_sd_market;

# COMMAND ----------

# San Diego Rankings Dashboard - ranks pre-computed in gold_sd_rankings
sd_rank = spark.table("gold_sd_rankings").toPandas()

if not sd_rank.empty:
    n = int(sd_rank["total_facilities"].iloc[0])
    metrics = [
        ("combined_charges",       "rank_charges",     "Combined Charges",    "$"),
        ("combined_medicare_pymt", "rank_pymt",        "Medicare Payments",   "$"),
        ("inp_discharges",         "rank_discharges",  "Inpatient Discharges","#"),
        ("out_services",           "rank_services",    "Outpatient Services", "#"),
        ("reimb_rate",             "rank_reimb_rate",   "Reimb. Rate (Pymt/Charge)", "%"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"San Diego Market Rankings ({n} facilities)", fontsize=15, fontweight="bold")

    for idx, (val_col, rank_col, title, fmt) in enumerate(metrics):
        ax = axes[idx // 3][idx % 3]
        sorted_df = sd_rank.sort_values(val_col, ascending=True)
        colors = ["#0072CE" if f == "Y" else "#CCCCCC"
                  for f in sorted_df["kaiser_flag"]]
        short_names = [n[:30] + "..." if len(n) > 30 else n for n in sorted_df["provider_name"]]
        bars = ax.barh(range(len(sorted_df)), sorted_df[val_col].values, color=colors)

        for i, (val, flag) in enumerate(zip(sorted_df[val_col].values, sorted_df["kaiser_flag"].values)):
            if fmt == "$":
                lbl = f"${val:,.0f}" if val < 1e6 else f"${val/1e6:,.1f}M"
            elif fmt == "#":
                lbl = f"{int(val):,}"
            elif fmt == "%":
                lbl = f"{val:.1%}"
            else:
                lbl = str(val)
            ax.text(val, i, f" {lbl}", va="center", fontsize=8,
                    fontweight="bold" if flag == "Y" else "normal")

        ax.set_yticks(range(len(sorted_df)))
        ax.set_yticklabels(short_names, fontsize=8)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[1][2].axis("off")
    plt.tight_layout()
    display(fig)
    plt.close(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Pipeline summary

# COMMAND ----------

print(f"{'TABLE':<35} {'ROWS':>10}")
print("-" * 47)
for t in spark.sql("SHOW TABLES").collect():
    if not t.tableName.startswith("_"):
        n = spark.table(t.tableName).count()
        print(f"{t.tableName:<35} {n:>10,}")