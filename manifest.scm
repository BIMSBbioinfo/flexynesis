;; Use this to set up a development environment.
;;   guix shell -m manifest.scm
;;
;; Optionally, add "--pure" or "--container" for more isolation.

(define %packages
  (list "python-matplotlib"
        "python-numpy"
        "python-pandas"
        "python-pytorch"
        "python-pytorch-lightning"
        "python-pyyaml"
        "python-scikit-optimize"
        "python-scipy"
        "python-seaborn"
        "python-torchvision"
        "python-tqdm"
        "python-umap-learn"))
(define %dev-packages
  (list "python-pytest"))

(specifications->manifest
 (cons "python" (append %packages %dev-packages)))
