-- MySQL dump 10.13  Distrib 8.0.42, for Linux (x86_64)
--
-- Host: localhost    Database: sprintsense
-- ------------------------------------------------------
-- Server version	8.0.42

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `users`
--

DROP TABLE IF EXISTS `users`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `users` (
  `id` int NOT NULL AUTO_INCREMENT,
  `provider` varchar(255) NOT NULL,
  `api_key` varchar(255) NOT NULL,
  `account_id` varchar(255) NOT NULL,
  `adapter_path` varchar(512) DEFAULT NULL,
  `adapter_created_at` timestamp NULL DEFAULT NULL,
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `predicted_velocity` float DEFAULT NULL,
  `predicted_sprint_duration_days` float DEFAULT NULL,
  `predicted_finished_on_time` tinyint(1) DEFAULT NULL,
  `predicted_duration_days` float DEFAULT NULL,
  `prediction_created_at` timestamp NULL DEFAULT NULL,
  `confidence_interval` float DEFAULT NULL,
  `risk_flag` tinyint(1) DEFAULT NULL,
  `auth_token` varchar(512) DEFAULT NULL,
  `llm_prediction` text,
  PRIMARY KEY (`id`),
  UNIQUE KEY `auth_token` (`auth_token`),
  UNIQUE KEY `auth_token_2` (`auth_token`)
) ENGINE=InnoDB AUTO_INCREMENT=4 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `users`
--

LOCK TABLES `users` WRITE;
/*!40000 ALTER TABLE `users` DISABLE KEYS */;
INSERT INTO `users` VALUES (3,'openai','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','712020:2703dfcd-47f8-4f5a-9a10-c26e88c7221e','/home/romankreiner/Documents/Hackathon/SprintSense/finished_models/dense/adapter_4c02a6cab006a0dbdafaadc1aeda1de2377f0e8c244737820b264dff83fb0b37.pth','2025-11-21 10:12:19','2025-11-21 09:41:59',31,17,1,NULL,'2025-11-21 10:30:18',NULL,NULL,'4c02a6cab006a0dbdafaadc1aeda1de2377f0e8c244737820b264dff83fb0b37','<h2>Executive Summary</h2>\n<p>\nThe LSTM model predicts a dramatic velocity jump from a consistent historical range of 1–2 to 31 story points, with a modest increase in sprint duration to 17 days. This forecast is an extreme statistical outlier, highly implausible given the single-member team\'s previous throughput. Blindly adopting these targets would introduce major delivery and planning risk.\n</p>\n\n<h2>Detailed Metric Analysis</h2>\n<ul>\n  <li>\n    <b>Velocity:</b> Predicted at 31 vs. historical average of ~1.7 (range: 1–2). This is an outlier exceeding 15x normal output. With unchanged team size (1), such a leap is not justified by historical inputs. No indication of process, resourcing, or input changes to support the surge.\n  </li>\n  <li>\n    <b>Sprint duration:</b> Predicted at 17 days, which is only slightly higher than the prior 14-day sprints and shorter than several 28-day sprints. There is no historic pattern associating similar sprint durations with massive velocity jumps.\n  </li>\n  <li>\n    <b>Finished on time:</b> Model predicts \"on time\" delivery, but given the unprecedented velocity target with no process or capacity changes evident, this is highly unreliable.\n  </li>\n  <li>\n    <b>Historical feature stability:</b> Team size, issues per sprint, and story points per member have all remained flat. No evidence of scaling that would explain the prediction.\n  </li>\n</ul>\n\n<h2>Risks & Red Flags</h2>\n<ul>\n  <li>\n    <b>Extreme overcommitment risk:</b> Setting velocity at 31 for a team historically achieving 2 will almost certainly result in failure to deliver, increased context-switching, and morale damage.\n  </li>\n  <li>\n    <b>Statistical anomaly:</b> LSTM prediction is a clear outlier and should be considered a model error or a misinterpretation of context/features.\n  </li>\n  <li>\n    <b>Process volatility:</b> No historic volatility or scaling supports such a leap. Following this target would break cadence and transparency.\n  </li>\n  <li>\n    <b>Potential workflow breakdown:</b> Delivery processes and capacity planning will be undermined by adopting such unrealistic targets.\n  </li>\n</ul>\n\n<h2>Improvement Recommendations</h2>\n<ul>\n  <li>\n    <b>Set realistic velocity target:</b> Cap next sprint velocity at or near recent actuals (2 story points) until validated change in team or process occurs.\n  </li>\n  <li>\n    <b>Review modeling assumptions:</b> Audit LSTM inputs and features—ensure future forecasts are constraint-aware and reflect operational reality.\n  </li>\n  <li>\n    <b>Capacity planning check:</b> Confirm no sudden increases in team size, automation, or scope simplification have been overlooked.\n  </li>\n  <li>\n    <b>Gradually scale throughput:</b> If aiming to improve, pilot a small, incremental velocity increase (e.g., from 2 to 3), but monitor closely.\n  </li>\n</ul>\n\n<h2>Additional Insights</h2>\n<ul>\n  <li>\n    Team throughput has shown remarkable stability, suggesting a mature but capacity-limited workflow. Sudden, large fluctuations in targets should be validated with hard resourcing or process changes, not solely on model output.\n  </li>\n</ul>\n');
/*!40000 ALTER TABLE `users` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2025-12-04 15:49:26
