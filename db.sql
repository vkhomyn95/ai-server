-- Create table tariff if does not exists
CREATE TABLE IF NOT EXISTS `tariff` (
    `id` int(11) NOT NULL AUTO_INCREMENT,
    `created_date` datetime DEFAULT NULL,
    `updated_date` datetime DEFAULT NULL,
    `request` bit(1) NOT NULL,
    `request_limit` int(11) NOT NULL,
    `request_size` int(11) DEFAULT 0,
    `audio` bit(1) NOT NULL,
    `audio_limit` int(11) NOT NULL,
    `audio_size` int(11) DEFAULT 0,
    PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
-- Create table user if does not exists
CREATE TABLE IF NOT EXISTS `user` (
    `id` int(11) NOT NULL AUTO_INCREMENT,
    `created_date` datetime DEFAULT NULL,
    `updated_date` datetime DEFAULT NULL,
    `first_name` varchar(128) DEFAULT NULL,
    `last_name` varchar(128) DEFAULT NULL,
    `email` varchar(255) DEFAULT NULL,
    `phone` varchar(128) DEFAULT NULL,
    `api_key` varchar(255) DEFAULT NULL,
    `secret_key` varchar(255) DEFAULT NULL,
    `audience` varchar(255) DEFAULT NULL,
    `tariff_id` int(11) DEFAULT NULL,
    PRIMARY KEY (`id`),
    KEY `fk_user_tariff` (`tariff_id`),
    CONSTRAINT `fk_user_tariff` FOREIGN KEY (`tariff_id`) REFERENCES `tariff` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
-- Create table recognition if does not exists
CREATE TABLE IF NOT EXISTS `recognition` (
    `id` int(11) NOT NULL AUTO_INCREMENT,
    `created_date` datetime(3) DEFAULT NULL,
    `final` bit(1) NOT NULL,
    `request_uuid` varchar(128) DEFAULT NULL,
    `audio_uuid` varchar(128) DEFAULT NULL,
    `confidence` int(11) NOT NULL,
    `prediction` varchar(64) DEFAULT NULL,
    `extension` varchar(64) DEFAULT NULL,
    `user_id` int(11) DEFAULT NULL,
    PRIMARY KEY (`id`),
    KEY `fk_recognition_user` (`user_id`),
    CONSTRAINT `fk_recognition_user` FOREIGN KEY (`user_id`) REFERENCES `user` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;