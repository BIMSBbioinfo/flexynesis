;; Use this to set up a development environment.
;;   guix time-machine -C channels.scm -- shell -m manifest.scm
;;
;; Optionally, add "--pure" or "--container" for more isolation.

(import (guix packages)
        (guix profiles))

(define replace-cpu-torch-with-gpu-torch
  (if (getenv "FLEXYNESIS_USE_CPU")
      identity  ;don't do anything
      (let ((cpu-torch (specification->package "python-pytorch"))
            (gpu-torch (specification->package "python-pytorch-with-cuda12")))
        (package-input-rewriting `((,cpu-torch . ,gpu-torch))))))

(define %packages
  (map replace-cpu-torch-with-gpu-torch
       (map specification->package
            (list "python-captum"
                  "python-geomloss"
                  "python-ipykernel"
                  "python-ipywidgets"
                  "python-lifelines"
                  "python-louvain"
                  "python-matplotlib"
                  "python-numpy"
                  "python-pandas"
                  "python-papermill"
                  "python-pot"
                  "python-pytorch"
                  "python-pytorch-geometric"
                  "python-pytorch-lightning"
                  "python-pyyaml"
                  "python-rich"
                  "python-scikit-optimize"
                  "python-scikit-survival"
                  "python-scipy"
                  "python-seaborn"
                  "python-torchvision"
                  "python-tqdm"
                  "python-umap-learn"
                  "python-xgboost"))))
(define %dev-packages
  (map specification->package
       (list "python-pytest")))

(packages->manifest
 (cons (specification->package "python")
       (append %packages %dev-packages)))
