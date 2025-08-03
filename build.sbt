name := "regression_metrics"

version := "0.1"

scalaVersion := "2.13.16"





libraryDependencies +="org.apache.logging.log4j" % "log4j-core" % "2.24.3"
libraryDependencies +=  "org.apache.logging.log4j" % "log4j-api" % "2.24.3"
libraryDependencies +=  "com.google.genai" % "google-genai" % "0.3.0"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "3.5.1"
libraryDependencies += "org.apache.spark" %% "spark-sql" % "3.5.1"

  
dependencyOverrides += "org.scala-lang.modules" %% "scala-parser-combinators" % "2.3.0"
dependencyOverrides += "com.fasterxml.jackson.core" % "jackson-databind" % "2.15.2"

fork := true
javaOptions ++= Seq(
  "--add-opens", "java.base/java.nio=ALL-UNNAMED",
  "--add-opens", "java.base/java.net=ALL-UNNAMED",
  "--add-opens", "java.base/java.lang=ALL-UNNAMED",
  "--add-opens", "java.base/java.util=ALL-UNNAMED",
  "--add-opens", "java.base/java.util.concurrent=ALL-UNNAMED"
)
