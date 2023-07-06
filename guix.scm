;; Use this to build a Guix package from the current git checkout.
;; Note that uncommitted changes will not be included!

;; Use like this:
;;   guix time-machine -C channels.scm -- build -f guix.scm

(use-modules (guix build-system pyproject)
             (guix git)
             (guix packages)
             (guix licenses)
             (gnu packages))

(include "manifest.scm")

(package
  (name "flexynesis")
  (version "0.1.0")
  (source (git-checkout (url (dirname (current-filename)))))
  (build-system pyproject-build-system)
  (arguments
   (list
    #:phases
    '(modify-phases %standard-phases
       (add-before 'check 'set-numba-cache-dir
         (lambda _
           (setenv "NUMBA_CACHE_DIR" "/tmp"))))))
  (propagated-inputs %packages)
  (native-inputs %dev-packages)
  (home-page "https://github.com/BIMSBbioinfo/flexynesis")
  (synopsis "Multi-omics bulk sequencing data integration suite")
  (description "This is a deep-learning based multi-omics bulk
sequencing data integration suite with a focus on (pre-)clinical
endpoint prediction.")
  (license #f))
