import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.SparkContext._

import scala.util.Random


/**
 * Created by xd on 2014/10/05.
 */
object Application {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Simple Application").setMaster("local")
    implicit val sc = new SparkContext(conf)

    wordcount()
    stats()
    logistic()
    decisionTree()
  }


  def stats()(implicit sc: SparkContext): Unit = {
    val vecs = loadIris map { v =>
      Vectors.dense(v._1, v._2, v._3, v._4)
    }

    val summary = Statistics.colStats(vecs)
    println(s"min: ${summary.min}")
    println(s"max: ${summary.max}")
    println(s"mean: ${summary.mean}")
    println(s"var: ${summary.variance}")
  }

  def logistic()(implicit sc: SparkContext): Unit = {
    val vecs = loadIris map { v =>
      val label =
        if (v._5 == "Iris-virginica") 0.0
        else 1.0
      LabeledPoint(label, Vectors.dense(v._1, v._2, v._3, v._4))
    }
    val splits = vecs.randomSplit(Array(0.8, 0.2))
    val train = splits(0).cache()
    val test = splits(1)

    val logi = LogisticRegressionWithSGD.train(train, 100)
    logi.clearThreshold()
    test foreach { point =>
      val pred = logi.predict(point.features)
      println(s"${point.label}:$pred")
    }
  }

  def decisionTree()(implicit sc: SparkContext): Unit = {
    val vecs = loadIris map { v =>
      val label = v._5 match {
        case "Iris-setosa" => 0
        case "Iris-versicolor" => 1
        case "Iris-virginica" => 2
        case _ => -1
      }
      LabeledPoint(label, Vectors.dense(v._1, v._2, v._3, v._4))
    }

    val splits = vecs.randomSplit(Array(0.8, 0.2))
    val train = splits(0).cache()
    val test = splits(1)

    val model = DecisionTree.trainClassifier(train, 3, Map[Int, Int](), "gini", 2, 150)
    test foreach { point =>
      val pred = model.predict(point.features)
      println(s"${point.label}:$pred")
    }
  }

  def wordcount()(implicit sc: SparkContext): Unit = {
    sc.parallelize(FileUtil.load("ihaveadream")).
      flatMap(_.split(" ")).
      filter(_ != "" ).
      map(w => (w.toLowerCase, 1)).
      reduceByKey((a, b) => a + b).
      collect(). // ここまでRDDとしての処理
      sortBy(_._2).
      reverse.
      take(20).
      foreach { case (word, count) =>
      println(s"$word: $count")
    }
  }

  def loadIris(implicit sc: SparkContext) = {
    sc.parallelize(Random.shuffle( FileUtil.load("iris.data") map { line =>
      line.split(",") match {
        case v => (v(0).toFloat, v(1).toFloat, v(2).toFloat, v(3).toFloat, v(4))
      }
    }))
  }
}