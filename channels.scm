(list (channel
       (name 'guix)
       (url "https://git.savannah.gnu.org/git/guix.git")
       (branch "master")
       (commit
        "da8afaa65fe30ae59e1bedbbb231490ad01c013c")
       (introduction
        (make-channel-introduction
         "9edb3f66fd807b096b48283debdcddccfea34bad"
         (openpgp-fingerprint
          "BBB0 2DDF 2CEA F6A8 0D1D  E643 A2A0 6DF2 A33A 54FA"))))
      (channel
       (name 'guix-science)
       (url "https://codeberg.org/guix-science/guix-science.git")
       (commit
        "e79f07d6ae81721e62d7cce78378f3ec49ff4efd")
       (introduction
        (make-channel-introduction
         "b1fe5aaff3ab48e798a4cce02f0212bc91f423dc"
         (openpgp-fingerprint
          "CA4F 8CF4 37D7 478F DA05  5FD4 4213 7701 1A37 8446"))))
      (channel
       (name 'guix-science-nonfree)
       (url "https://codeberg.org/guix-science/guix-science-nonfree.git")
       (commit
        "446626ab1ca977b9278f431da8dde9ec8cf36457")
       (introduction
        (make-channel-introduction
         "58661b110325fd5d9b40e6f0177cc486a615817e"
         (openpgp-fingerprint
          "CA4F 8CF4 37D7 478F DA05  5FD4 4213 7701 1A37 8446")))))
