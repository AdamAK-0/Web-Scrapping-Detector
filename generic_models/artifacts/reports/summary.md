# Generic Website Detector Results

This is a separate model family from the live-site thesis models.

## Data

- Simulated websites: 72
- Simulated sessions: 6480
- Label counts: {'human': 3240, 'bot': 3240}
- Families: {'bfs': 466, 'browser_like': 477, 'comparison': 687, 'deep_harvest': 421, 'dfs': 483, 'distracted': 603, 'exploratory': 644, 'focused': 453, 'goal_oriented': 665, 'linear': 470, 'noisy': 470, 'support_seeking': 641}

## Generic Leave-Site-Out Leaderboard

```text
            model_name                          dataset  threshold  train_sites  validation_sites  test_sites  accuracy  precision   recall       f1  roc_auc   pr_auc
hist_gradient_boosting generic_multisite_leave_site_out       0.55           43                14          15  0.986939   0.989670 0.986056 0.987860 0.998737 0.999041
              lightgbm generic_multisite_leave_site_out       0.60           43                14          15  0.986223   0.989330 0.985060 0.987190 0.999044 0.999234
         random_forest generic_multisite_leave_site_out       0.50           43                14          15  0.986044   0.988674 0.985392 0.987030 0.999197 0.999353
               xgboost generic_multisite_leave_site_out       0.60           43                14          15  0.986044   0.989653 0.984396 0.987017 0.999089 0.999262
              catboost generic_multisite_leave_site_out       0.30           43                14          15  0.985686   0.981287 0.992364 0.986794 0.999267 0.999414
           extra_trees generic_multisite_leave_site_out       0.45           43                14          15  0.983897   0.984095 0.986056 0.985075 0.998751 0.999026
   logistic_regression generic_multisite_leave_site_out       0.45           43                14          15  0.982823   0.984064 0.984064 0.984064 0.998526 0.998739
        calibrated_svm generic_multisite_leave_site_out       0.60           43                14          15  0.982108   0.987609 0.979084 0.983328 0.998651 0.998849
```

## Prefix Metrics

```text
            model_name  prefix_len  accuracy  precision   recall       f1  roc_auc   pr_auc
        calibrated_svm           3  0.945926   0.958841 0.931852 0.945154 0.990374 0.990454
        calibrated_svm           5  0.986667   0.992504 0.980741 0.986587 0.999173 0.999235
        calibrated_svm          10  0.995788   0.995413 0.996937 0.996174 0.999946 0.999955
        calibrated_svm          15  0.996907   0.996422 0.998208 0.997314 0.999974 0.999981
        calibrated_svm          20  0.998634   1.000000 0.997783 0.998890 1.000000 1.000000
              catboost           3  0.960000   0.944206 0.977778 0.960699 0.995768 0.995970
              catboost           5  0.988148   0.982430 0.994074 0.988218 0.998819 0.999064
              catboost          10  0.994103   0.992378 0.996937 0.994652 0.999911 0.999928
              catboost          15  0.997938   0.998208 0.998208 0.998208 0.999987 0.999990
              catboost          20  0.998634   1.000000 0.997783 0.998890 1.000000 1.000000
           extra_trees           3  0.956296   0.948980 0.964444 0.956650 0.993146 0.993991
           extra_trees           5  0.985926   0.988095 0.983704 0.985895 0.999023 0.999051
           extra_trees          10  0.992418   0.993865 0.992343 0.993103 0.999848 0.999875
           extra_trees          15  0.996907   0.998205 0.996416 0.997309 0.999957 0.999968
           extra_trees          20  1.000000   1.000000 1.000000 1.000000 1.000000 1.000000
hist_gradient_boosting           3  0.970370   0.971768 0.968889 0.970326 0.994043 0.994657
hist_gradient_boosting           5  0.983704   0.986587 0.980741 0.983655 0.998310 0.998659
hist_gradient_boosting          10  0.993260   0.995392 0.992343 0.993865 0.999578 0.999695
hist_gradient_boosting          15  0.997938   1.000000 0.996416 0.998205 0.999765 0.999841
hist_gradient_boosting          20  0.998634   1.000000 0.997783 0.998890 0.999953 0.999971
              lightgbm           3  0.964444   0.968610 0.960000 0.964286 0.994469 0.994756
              lightgbm           5  0.984444   0.986607 0.982222 0.984410 0.998804 0.998990
              lightgbm          10  0.995788   0.996933 0.995406 0.996169 0.999857 0.999887
              lightgbm          15  0.997938   1.000000 0.996416 0.998205 0.999935 0.999953
              lightgbm          20  0.998634   1.000000 0.997783 0.998890 0.999968 0.999981
   logistic_regression           3  0.948148   0.948148 0.948148 0.948148 0.990007 0.989953
   logistic_regression           5  0.987407   0.988131 0.986667 0.987398 0.998951 0.999040
   logistic_regression          10  0.995788   0.995413 0.996937 0.996174 0.999905 0.999923
   logistic_regression          15  0.996907   0.996422 0.998208 0.997314 0.999965 0.999975
   logistic_regression          20  0.998634   1.000000 0.997783 0.998890 1.000000 1.000000
         random_forest           3  0.965185   0.963127 0.967407 0.965262 0.995142 0.995609
         random_forest           5  0.984444   0.989521 0.979259 0.984363 0.999355 0.999365
         random_forest          10  0.994103   0.996923 0.992343 0.994628 0.999877 0.999899
         random_forest          15  0.997938   1.000000 0.996416 0.998205 0.999983 0.999987
         random_forest          20  0.998634   1.000000 0.997783 0.998890 1.000000 1.000000
               xgboost           3  0.964444   0.971429 0.957037 0.964179 0.994607 0.994948
               xgboost           5  0.983704   0.985141 0.982222 0.983680 0.998872 0.999005
               xgboost          10  0.995788   0.996933 0.995406 0.996169 0.999854 0.999884
               xgboost          15  0.997938   1.000000 0.996416 0.998205 0.999948 0.999962
               xgboost          20  0.998634   1.000000 0.997783 0.998890 0.999992 0.999995
```

## Public Zenodo Benchmark

```text
                       dataset             model_name  threshold  feature_count  accuracy  precision   recall       f1  roc_auc   pr_auc
zenodo_public_session_features               lightgbm       0.80             35  0.979794   0.956222 0.944148 0.950147 0.995613 0.987203
zenodo_public_session_features          random_forest       0.55             35  0.978209   0.930679 0.965032 0.947544 0.996169 0.987583
zenodo_public_session_features                xgboost       0.45             35  0.978011   0.941799 0.950947 0.946351 0.995839 0.987584
zenodo_public_session_features hist_gradient_boosting       0.50             35  0.977813   0.946472 0.944633 0.945552 0.995610 0.987322
zenodo_public_session_features               catboost       0.75             35  0.975337   0.928098 0.952890 0.940331 0.995245 0.984606
zenodo_public_session_features            extra_trees       0.75             35  0.974544   0.939084 0.935891 0.937485 0.995241 0.984336
zenodo_public_session_features         calibrated_svm       0.30             35  0.949980   0.821074 0.965032 0.887252 0.985095 0.947525
zenodo_public_session_features    logistic_regression       0.70             35  0.949980   0.824833 0.958232 0.886542 0.983388 0.936759
```

## Models

Saved under `generic_models\artifacts\models`.

## Feature Coverage

The generic feature set includes coverage ratio, path entropy, revisit rate, depth distribution, branching decision patterns, inter-hop timing, entry/exit centrality, graph distance traveled, backtrack ratio, and structural roles such as hub/leaf/bridge visits.
