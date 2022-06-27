import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.evaluation.RegressionEvaluator


object RegresionLineal {
  def main(args: Array[String]) {
    val spark = SparkSession.builder().master("local[1]").appName("Regresion Lineal para Crimenes en la ciudad de Chicago").getOrCreate();
    val path = "Chicago_Crimes_2012_to_2017.csv" // Should be some file on your system
    /*
    val logData = sc.textFile(logFile,2).cache()
    val numAs = logData.filter(line => line.contains("a")).count()
    val numBs = logData.filter(line => line.contains("b")).count()
    println(s"Lines with a: $numAs, Lines with b: $numBs")
     */
    val schema = new StructType()
      .add("Number",LongType,true)
      .add("ID",LongType,true)
      .add("Case Number",StringType,true)
      .add("Date",StringType,true)
      .add("Block",StringType,true)
      .add("IUCR",IntegerType,true)
      .add("Primary Type",StringType,true)
      .add("Description",StringType,true)
      .add("Location Description",StringType,true)
      .add("Arrest",BooleanType,true)
      .add("Domestic",BooleanType,true)
      .add("Beat",IntegerType,true)
      .add("District",DoubleType,true)
      .add("Ward",DoubleType,true)
      .add("Community Area",FloatType,true)
      .add("FBI Code",StringType,true)
      .add("X Coordinate",DoubleType,true)
      .add("Y Coordinate",DoubleType,true)
      .add("Year",IntegerType,true)
      .add("Updated On",StringType,true)
      .add("Latitude",DoubleType,true)
      .add("Longitude",DoubleType,true)
      .add("Location",StringType,true)

    val df = spark.read.format("csv").option("header","true").schema(schema).load(path)
    df.printSchema()
    df.show()


    val featureCols= Array("Number","ID","Case Number","Date Block","IUCR","Primary Type","Description","Location Description","Arrest","Domestic","Beat","District","Ward","Community Area","FBI Code","X Coordinate","Y Coordinate","Year","Updated On","Latitude","Longitude","Location")

    val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")

    val dataDF = assembler.transform(df)
    dataDF.show(3)

    val Array(train, test) = dataDF1.randomSplit(Array(0.8, 0.2))
    println(train.count, rawData.count)
    train.show(3)
    //val assembler= new Vector[]()
    //assembler=new VectorAssembler().setInputCols(featureCols).setOutputCol("features")

    val lr= new LinearRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)

    //Fit the model
    val lrModel = lr.fit(train)

    // Print the coefficients and intercept for linear regression
    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

    // Summarize the model over the training set and print out some metrics
    val trainingSummary = lrModel.summary
    println(s"numIterations: ${trainingSummary.totalIterations}")
    println(s"objectiveHistory: [${trainingSummary.objectiveHistory.mkString(",")}]")
    trainingSummary.residuals.show()
    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
    println(s"r2: ${trainingSummary.r2}")

    val predictions = lrModel.transform(test)
    predictions.show()
    val eval = new RegressionEvaluator().setMetricName("rmse").setLabelCol("label").setPredictionCol("prediction")
    val rmse = eval.evaluate(predictions)

  }
}