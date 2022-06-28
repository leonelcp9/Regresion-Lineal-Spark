
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark import SparkConf
import pyspark.sql.functions as F
from pyspark.sql.types import StructType,StructField, StringType, IntegerType
from pyspark.sql import types as T

plt.style.use(style="seaborn")


spark = SparkSession.builder.appName("Indice de Crimenes en Chicago").getOrCreate()
spark

unpack_list_udf = F.udf(
    lambda x: list(set([item for sublist in x for item in sublist])),
    T.ArrayType(T.StringType()),
)

mean_udf = F.udf(lambda x: float(np.mean(x)), T.FloatType())

median_udf = F.udf(lambda x: float(np.median(x)), T.FloatType())

# schema = StructType([ \
#     StructField("firstname",StringType(),True), \
#     StructField("middlename",StringType(),True), \
#     StructField("lastname",StringType(),True), \
#     StructField("id", StringType(), True), \
#     StructField("gender", StringType(), True), \
#     StructField("salary", IntegerType(), True) \
#   ])

# df = spark.createDataFrame(data=data,schema=schema)
# df.printSchema()

df = spark.read.csv("./Chicago_Crimes_2012_to_2017.csv", inferSchema=True, header=True)

df = df.toDF(
    "_c0",
    "ID",
    "Case Number",
    "Date",
    "Block",
    "IUCR",
    "Primary Type",
    "Description",
    "Location Description",
    "Arrest",
    "Domestic",
    "Beat",
    "District",
    "Ward",
    "Community Area",
    "FBI Code",
    "X Coordinate",
    "Y Coordinate",
    "Year",
    "Updated On",
    "Latitude",
    "Longitude",
    "Location"
)

def procesando_datos(data, loc_descripcion, tipo_crimenes, arrestado):
    dato_locacion = {}

    assert data.groupBy("ID").count().count() == data.count()

    for tipo_crimen in tipo_crimenes:

        filtered = data.filter(data["Primary Type"] == tipo_crimen)
        if arrestado:
            filtered = filtered.filter(data["Arrest"] == "True")
        filtered = filtered.withColumn(
            "Date", F.from_unixtime(F.unix_timestamp("Date",'MM/dd/yyyy hh:mm:ss a'),'yyyy-MM-dd').cast('date')
        )

        dato_locacion[tipo_crimen] = {}
        for locacion in loc_descripcion:
            sub = (
                filtered.filter(filtered["Location Description"] == locacion)
                .groupBy("Date", F.window("Date", "30 days"))
                .agg(
                    F.expr("count('ID')").alias("Count"),
                )
            )

            sub = sub.select("Date", "window.*","Count").sort(F.asc("end"))

            sub = sub.groupBy("end").agg(
                F.expr("sum(Count)").alias("Count")
            )

            dato_locacion[tipo_crimen][locacion] = sub

    return dato_locacion


tipo_crimenes = ["THEFT", "ASSAULT", "ROBBERY", "STALKING", "BATTERY", "OTHER OFFENSE", "KIDNAPPING", "NARCOTICS"]
loc_descripcion = df.groupBy("Location Description").count().sort(F.desc("count")).collect()
loc_descripcion = [x["Location Description"] for x in loc_descripcion]

loc_descripcion[:20]
len(loc_descripcion)

dato_procesado = procesando_datos(df, loc_descripcion[:20], tipo_crimenes, True)
dato_procesado["BATTERY"]["SIDEWALK"].show(10)

import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category = matplotlib.cbook.mplDeprecation)

crimen = "BATTERY"
fig = plt.subplots(1, 1, sharex=True,figsize=(16,16))
for locacion in list(dato_procesado[crimen].keys())[1:6]:
    sub = dato_procesado[crimen][locacion].toPandas()
    plt.plot(sub.end,sub.Count, label = locacion)
    plt.ticklabel_format(style ='plain', axis ='y')
    plt.legend()

    dato_procesado["BATTERY"]["SIDEWALK"].select("Count").show(10)

from pyspark.sql.window import Window

sub = (
    dato_procesado["BATTERY"]["SIDEWALK"]
    .select(
        "Count",
        F.lead("Count")
        .over(Window().partitionBy().orderBy(F.col("end")))
        .alias("siguiente"),
        )
    .na.drop()
)
sub.show(10)

from pyspark.ml.feature import VectorAssembler

vectorAssembler = VectorAssembler(inputCols=["Count"], outputCol="features")
sub = vectorAssembler.transform(sub)

sub.select("features").show(10)

from pyspark.ml.regression import LinearRegression

reg = LinearRegression()
model = reg.fit(sub)

model.coefficients[0]
model.intercept

from  pyspark.ml.regression import LinearRegression
from pyspark.sql.window import Window

spark_lr = {}
for locacion in list(dato_procesado["BATTERY"].keys())[:10]:
    sub = (
        dato_procesado["BATTERY"][locacion]
        .select(
            "Count",
            F.lead("Count")
            .over(Window().partitionBy().orderBy(F.col("end")))
            .alias("label"),
            )
        .na.drop()
    )
    vectorAssembler = VectorAssembler(inputCols=["Count"], outputCol="features")
    sub = vectorAssembler.transform(sub)

    reg = LinearRegression()
    model = reg.fit(sub)
    spark_lr[locacion] = model.coefficients[0]

desc_spark_lr = {k: v for k, v in sorted(spark_lr.items(), key=lambda item: item[1], reverse=True)}

print(desc_spark_lr)
dato_procesado["BATTERY"]["SIDEWALK"].select("Count").show(10)


from pyspark.sql.window import Window

sub = (
    dato_procesado["BATTERY"]["SIDEWALK"]
    .select(
        "Count",
        F.lead("Count")
        .over(Window().partitionBy().orderBy(F.col("end")))
        .alias("label"),
        )
    .na.drop()
)
sub.show(10)

from pyspark.ml.feature import VectorAssembler

vectorAssembler = VectorAssembler(inputCols=["Count"], outputCol="features")
sub = vectorAssembler.transform(sub)

sub.select("features").show(10)

from pyspark.ml.regression import LinearRegression

reg = LinearRegression()
model = reg.fit(sub)

model.coefficients[0]
model.intercept

from pyspark.ml.regression import LinearRegression
from pyspark.sql.window import Window

spark_lr = {}
for locacion in list(dato_procesado["BATTERY"].keys())[:10]:
    sub = (
        dato_procesado["BATTERY"][locacion]
        .select(
            "Count",
            F.lead("Count")
            .over(Window().partitionBy().orderBy(F.col("end")))
            .alias("label"),
            )
        .na.drop()
    )
    vectorAssembler = VectorAssembler(inputCols=["Count"], outputCol="features")
    sub = vectorAssembler.transform(sub)

    reg = LinearRegression()
    model = reg.fit(sub)
    spark_lr[locacion] = model.coefficients[0]

desc_spark_lr = {k: v for k, v in sorted(spark_lr.items(), key=lambda item: item[1], reverse=True)}

print(desc_spark_lr)







