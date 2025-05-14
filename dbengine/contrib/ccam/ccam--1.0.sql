-- ccam--1.0.sql
CREATE OR REPLACE FUNCTION ccam_tableam_handler(internal)
RETURNS table_am_handler AS 'ccam', 'ccam_tableam_handler'
LANGUAGE C STRICT;

CREATE ACCESS METHOD ccam TYPE TABLE HANDLER ccam_tableam_handler;
