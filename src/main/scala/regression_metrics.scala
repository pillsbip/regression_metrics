package regression_metrics


import scala.util.{Failure, Success, Try}
import java.nio.file.{Files, Paths}
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._






object CalculateRegressionMetrics {

 def main(args: Array[String]): Unit = {
  


   try{



// Load training data in LIBSVM format

   val spark = SparkSession.builder()
      .appName("Regression metrics")
      .master("local[*]")
      .getOrCreate()
      
    // Log level
    spark.sparkContext.setLogLevel("WARN")

//val conf = new SparkConf().setAppName("LibSVM Loader").setMaster("local[*]")
val sc = spark.sparkContext



// Now you can load the data

val data = MLUtils.loadLibSVMFile(sc, "src/main/scala/resources/classification_experiment_data.txt")



// Split data into training (60%) and test (40%)
val Array(training, test) = data.randomSplit(Array(0.6, 0.4), seed = 11L)
training.cache()

// Run training algorithm to build the model
val model = new LogisticRegressionWithLBFGS()
  .setNumClasses(3)
  .run(training)

// Compute raw scores on the test set
val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
  val prediction = model.predict(features)
  (prediction, label)
}


// Load the data stored in LIBSVM format as a DataFrame.
val dataMLP = spark.read.format("libsvm")
  .load("src/main/scala/resources/classification_experiment_data.txt")

// Split the data into train and test
val splits = dataMLP.randomSplit(Array(0.6, 0.4), seed = 1234L)
val trainMLP = splits(0)
val testMLP = splits(1)

// specify layers for the neural network:
// input layer of size 4 (features), two intermediate of size 5 and 4
// and output of size 3 (classes)
val layers = Array[Int](4, 5, 4, 3)

// create the trainer and set its parameters
val trainerMLP = new MultilayerPerceptronClassifier()
  .setLayers(layers)
  .setBlockSize(128)
  .setSeed(1234L)
  .setMaxIter(100)

// train the model
val modelMLP = trainerMLP.fit(trainMLP)

// compute accuracy on the test set
val resultMLP = modelMLP.transform(testMLP)
val predictionAndLabelsMLP = resultMLP.select("prediction", "label")
val evaluatorMLP = new MulticlassClassificationEvaluator()
  .setMetricName("accuracy")

println(s"Test set accuracy = ${evaluatorMLP.evaluate(predictionAndLabelsMLP)}")

val predictionAndLabelsRDD = predictionAndLabelsMLP
  .select("prediction", "label")
  .rdd
  .map(row => (row.getDouble(0), row.getDouble(1)))

val metricsRDD = new MulticlassMetrics(predictionAndLabelsRDD)

// Confusion matrix
println("Confusion matrix:")
println(metricsRDD.confusionMatrix)

// Instantiate metrics object
val metrics = new MulticlassMetrics(predictionAndLabels)

// Confusion matrix
println("Confusion matrix:")
println(metrics.confusionMatrix)

// Overall Statistics
val accuracy = metrics.accuracy
println("Summary Statistics")
println(s"Accuracy = $accuracy")

// Precision by label
val labels = metrics.labels
labels.foreach { l =>
  println(s"Precision($l) = " + metrics.precision(l))
}

// Recall by label
labels.foreach { l =>
  println(s"Recall($l) = " + metrics.recall(l))
}

// False positive rate by label
labels.foreach { l =>
  println(s"FPR($l) = " + metrics.falsePositiveRate(l))
}

// F-measure by label
labels.foreach { l =>
  println(s"F1-Score($l) = " + metrics.fMeasure(l))
}

// Weighted stats
println(s"Weighted precision: ${metrics.weightedPrecision}")
println(s"Weighted recall: ${metrics.weightedRecall}")
println(s"Weighted F1 score: ${metrics.weightedFMeasure}")
println(s"Weighted false positive rate: ${metrics.weightedFalsePositiveRate}") 
   }catch {
  case e: Exception =>
    println(s"Error occurred: ${e.getMessage}")
    }
}
}