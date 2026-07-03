-- ============================================================================
-- CasJobs / SkyServer SQL query -- SDSS Data Release 17 (DR17)
-- ----------------------------------------------------------------------------
-- Selects the spectroscopically-confirmed star / galaxy / quasar sample used in
-- "Calibration Robustness of Photometric Star, Galaxy, and Quasar Classifiers
-- Under Magnitude Shift" and writes exactly the columns consumed by
-- src/calibration_experiment.py:
--
--     bestObjID, class, z,
--     dered_u, dered_g, dered_r, dered_i, dered_z,
--     modelMagErr_u, modelMagErr_g, modelMagErr_r, modelMagErr_i, modelMagErr_z
--
-- Selection (per paper Section II-A):
--   * spectroscopic ground truth from the clean SpecObj view (science primary),
--   * reliable redshift            -> zWarning = 0,
--   * matched to its PRIMARY, CLEAN photometric counterpart in PhotoObjAll via
--     bestObjID                    -> p.mode = 1 (primary) AND p.clean = 1,
--   * class restricted to STAR / GALAXY / QSO.
--
-- HOW TO RUN
--   SDSS CasJobs (https://skyserver.sdss.org/casjobs/), context = DR17.
--   Submit as a long (batch) query; it returns ~5 x 10^5 rows. Export the
--   resulting table to CSV as data/sdss.csv, then:
--       python src/calibration_experiment.py --data data/sdss.csv --outdir results
--
-- NOTE: This file reconstructs the query faithfully from the paper's Methods;
-- the original .sql was not preserved alongside the exported catalog. Row counts
-- may differ by a handful from the paper (499,995) because the pipeline additionally
-- drops sources with non-finite or sentinel (e.g. -9999) magnitudes at load time.
-- ============================================================================

SELECT
    s.bestObjID,
    s.class,
    s.z,
    p.dered_u,
    p.dered_g,
    p.dered_r,
    p.dered_i,
    p.dered_z,
    p.modelMagErr_u,
    p.modelMagErr_g,
    p.modelMagErr_r,
    p.modelMagErr_i,
    p.modelMagErr_z
INTO mydb.sdss_sgq_dr17
FROM SpecObj AS s
JOIN PhotoObjAll AS p
       ON p.objID = s.bestObjID
WHERE s.zWarning = 0
  AND s.class IN ('STAR', 'GALAXY', 'QSO')
  AND p.mode  = 1     -- primary photometric object
  AND p.clean = 1     -- clean photometry flag
