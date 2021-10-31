package eig

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// zckgqr tests
// Zggqrf: gqr factorization for N-by-M matrix A and N-by-P matrix B,
// Zggrqf: grq factorization for M-by-N matrix A and P-by-N matrix B.
func zckgqr(nm int, mval []int, np int, pval []int, nn int, nval []int, nmats int, iseed []int, thresh float64, nmax int, a, af, aq, ar, taua, b, bf, bz, bt, bwk, taub, work *mat.CVector, rwork *mat.Vector) (nfail, nrun int, err error) {
	var firstt bool
	var dista, distb, _type byte
	var anorm, bnorm, cndnma, cndnmb float64
	var i, iinfo, im, imat, in, ip, kla, klb, kua, kub, lda, ldb, lwork, m, modea, modeb, n, nt, ntypes, p int
	dotype := make([]bool, 8)
	result := vf(7)

	// ntests = 7
	ntypes = 8

	//     Initialize constants.
	path := "Gqr"
	nrun = 0
	nfail = 0
	firstt = true
	alareq(nmats, &dotype)
	lda = nmax
	ldb = nmax
	lwork = nmax * nmax

	//     Do for each value of M in MVAL.
	for im = 1; im <= nm; im++ {
		m = mval[im-1]

		//        Do for each value of P in PVAL.
		for ip = 1; ip <= np; ip++ {
			p = pval[ip-1]

			//           Do for each value of N in NVAL.
			for in = 1; in <= nn; in++ {
				n = nval[in-1]

				for imat = 1; imat <= ntypes; imat++ {
					//                 Do the tests only if DOTYPE( IMAT ) is true.
					if !dotype[imat-1] {
						goto label30
					}

					//                 Test Zggrqf
					//
					//                 Set up parameters with DLATB9 and generate test
					//                 matrices A and B with Zlatms.
					_type, kla, kua, klb, kub, anorm, bnorm, modea, modeb, cndnma, cndnmb, dista, distb = dlatb9("Grq", imat, m, p, n)

					if err = matgen.Zlatms(m, n, dista, &iseed, _type, rwork, modea, cndnma, anorm, kla, kua, 'N', a.CMatrix(lda, opts), work); err != nil {
						fmt.Printf(" Zlatms in zckgqr:    INFO = %5d\n", iinfo)
						err = fmt.Errorf("iinfo=%v", abs(iinfo))
						goto label30
					}

					if err = matgen.Zlatms(p, n, distb, &iseed, _type, rwork, modeb, cndnmb, bnorm, klb, kub, 'N', b.CMatrix(ldb, opts), work); err != nil {
						fmt.Printf(" Zlatms in zckgqr:    INFO = %5d\n", iinfo)
						err = fmt.Errorf("iinfo=%v", abs(iinfo))
						goto label30
					}

					nt = 4

					zgrqts(m, p, n, a.CMatrix(lda, opts), af.CMatrix(lda, opts), aq.CMatrix(lda, opts), ar.CMatrix(lda, opts), taua, b.CMatrix(ldb, opts), bf.CMatrix(ldb, opts), bz.CMatrix(ldb, opts), bt.CMatrix(ldb, opts), bwk.CMatrix(ldb, opts), taub, work, lwork, rwork, result)

					//                 Print information about the tests that did not
					//                 pass the threshold.
					for i = 1; i <= nt; i++ {
						if result.Get(i-1) >= thresh {
							if nfail == 0 && firstt {
								firstt = false
								alahdg("Grq")
							}
							fmt.Printf(" m=%4d p=%4d, n=%4d, _type %2d, test %2d, ratio=%13.6f\n", m, p, n, imat, i, result.Get(i-1))
							err = fmt.Errorf(" m=%4d p=%4d, n=%4d, _type %2d, test %2d, ratio=%13.6f\n", m, p, n, imat, i, result.Get(i-1))
							nfail++
						}
					}
					nrun = nrun + nt

					//                 Test Zggqrf
					//
					//                 Set up parameters with DLATB9 and generate test
					//                 matrices A and B with Zlatms.
					_type, kla, kua, klb, kub, anorm, bnorm, modea, modeb, cndnma, cndnmb, dista, distb = dlatb9("Gqr", imat, m, p, n)

					if err = matgen.Zlatms(n, m, dista, &iseed, _type, rwork, modea, cndnma, anorm, kla, kua, 'N', a.CMatrix(lda, opts), work); err != nil {
						fmt.Printf(" Zlatms in zckgqr:    info = %5d\n", iinfo)
						err = fmt.Errorf("iinfo=%v", abs(iinfo))
						goto label30
					}

					if err = matgen.Zlatms(n, p, distb, &iseed, _type, rwork, modea, cndnma, bnorm, klb, kub, 'N', b.CMatrix(ldb, opts), work); err != nil {
						fmt.Printf(" Zlatms in zckgqr:    info = %5d\n", iinfo)
						err = fmt.Errorf("iinfo=%v", abs(iinfo))
						goto label30
					}

					nt = 4

					zgqrts(n, m, p, a.CMatrix(lda, opts), af.CMatrix(lda, opts), aq.CMatrix(lda, opts), ar.CMatrix(lda, opts), taua, b.CMatrix(ldb, opts), bf.CMatrix(ldb, opts), bz.CMatrix(ldb, opts), bt.CMatrix(ldb, opts), bwk.CMatrix(ldb, opts), taub, work, lwork, rwork, result)

					//                 Print information about the tests that did not
					//                 pass the threshold.
					for i = 1; i <= nt; i++ {
						if result.Get(i-1) >= thresh {
							if nfail == 0 && firstt {
								firstt = false
								alahdg(path)
							}
							fmt.Printf(" n=%4d m=%4d, p=%4d, _type %2d, test %2d, ratio=%13.6f\n", n, m, p, imat, i, result.Get(i-1))
							err = fmt.Errorf(" n=%4d m=%4d, p=%4d, _type %2d, test %2d, ratio=%13.6f\n", n, m, p, imat, i, result.Get(i-1))
							nfail++
						}
					}
					nrun = nrun + nt

				label30:
				}
			}
		}
	}

	//     Print a summary of the results.
	// alasum(path, nfail, nrun, 0)

	return
}
