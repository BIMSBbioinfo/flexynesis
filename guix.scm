;; Use this to build a Guix package from the current git checkout.
;; Note that uncommitted changes will not be included!

;; Use like this:
;;   guix time-machine -C channels.scm -- build -f guix.scm

(use-modules (guix build-system pyproject)
             (guix git)
             (guix gexp)
             (guix packages)
             (guix licenses)
             (gnu packages))

(include "manifest.scm")

(package
  (name "flexynesis")
  (version "0.2.10")
  (source (git-checkout (url (dirname (current-filename)))))
  (build-system pyproject-build-system)
  (arguments
   (list
    #:phases
    #~(modify-phases %standard-phases
        (add-after 'unpack 'disable-doctests
          (lambda _
            ;; Disable doctests because they are broken.  See
            ;; https://github.com/BIMSBbioinfo/flexynesis/issues/93
            (substitute* "pyproject.toml"
              ((".*--doctest-modules.*") ""))))
        (add-before 'check 'set-numba-cache-dir
          (lambda _
            (setenv "NUMBA_CACHE_DIR" "/tmp")))
        (add-after 'wrap 'isolate-python
          (lambda _
            (substitute* (string-append #$output "/bin/.flexynesis-real")
              (("/bin/python") "/bin/python -I")))))))
  (propagated-inputs %packages)
  (native-inputs %dev-packages)
  (home-page "https://github.com/BIMSBbioinfo/flexynesis")
  (synopsis "Multi-omics bulk sequencing data integration suite")
  (description "This is a deep-learning based multi-omics bulk
sequencing data integration suite with a focus on (pre-)clinical
endpoint prediction.")
  (license #f))
