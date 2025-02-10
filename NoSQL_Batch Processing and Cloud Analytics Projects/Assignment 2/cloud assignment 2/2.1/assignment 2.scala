// Databricks notebook source
val TextFile = spark.sparkContext.textFile("/FileStore/tables/Text_Corpus/*.text")

// COMMAND ----------

TextFile.count()

// COMMAND ----------

val antibioticsCount = TextFile.filter(line => line.contains("antibiotics")).count()


// COMMAND ----------

val linesWithAntibioticsDF = TextFile.filter(line => line.contains("antibiotics")).toDF("lines")

// Show the DataFrame containing lines with "antibiotics"
linesWithAntibioticsDF.show(false)

// COMMAND ----------

val patientAdmittedCount = TextFile.filter(line => line.contains("patient"))
                                   .filter(line => line.contains("admitted"))
patientAdmittedCount.count()


// COMMAND ----------

// Filter lines containing both "patient" and "admitted", and create a DataFrame
val linesWithPatientAdmittedDF = TextFile.filter(line => line.contains("patient"))
                                         .filter(line => line.contains("admitted")).toDF("lines")

// Show the DataFrame containing lines with both "patient" and "admitted"
linesWithPatientAdmittedDF.show(false)

// COMMAND ----------


