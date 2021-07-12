package eig

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// Dckgqr tests
// DGGQRF: GQR factorization for N-by-M matrix A and N-by-P matrix B,
// DGGRQF: GRQ factorization for M-by-N matrix A and P-by-N matrix B.
func Dckgqr(nm *int, mval *[]int, np *int, pval *[]int, nn *int, nval *[]int, nmats *int, iseed *[]int, thresh *float64, nmax *int, a, af, aq, ar *mat.Matrix, taua *mat.Vector, b, bf, bz, bt, bwk *mat.Matrix, taub, work, rwork *mat.Vector, nout *int, info *int, t *testing.T) {
	var firstt bool
	var dista, distb, _type byte
	var anorm, bnorm, cndnma, cndnmb float64
	var i, iinfo, im, imat, in, ip, kla, klb, kua, kub, lda, ldb, lwork, m, modea, modeb, n, nfail, nrun, nt, ntypes, p int

	dotype := make([]bool, 8)
	result := vf(7)

	ntypes = 8

	//     Initialize constants.
	path := []byte("GQR")
	(*info) = 0
	nrun = 0
	nfail = 0
	firstt = true
	Alareq(nmats, &dotype)
	lda = (*nmax)
	ldb = (*nmax)
	lwork = (*nmax) * (*nmax)

	//     Do for each value of M in MVAL.
	for im = 1; im <= (*nm); im++ {
		m = (*mval)[im-1]

		//        Do for each value of P in PVAL.
		for ip = 1; ip <= (*np); ip++ {
			p = (*pval)[ip-1]

			//           Do for each value of N in NVAL.
			for in = 1; in <= (*nn); in++ {
				n = (*nval)[in-1]

				for imat = 1; imat <= ntypes; imat++ {
					//                 Do the tests only if DOTYPE( IMAT ) is true.
					if !dotype[imat-1] {
						goto label30
					}

					//                 Test DGGRQF
					//
					//                 Set up parameters with DLATB9 and generate test
					//                 matrices A and B with DLATMS.
					Dlatb9([]byte("GRQ"), &imat, &m, &p, &n, &_type, &kla, &kua, &klb, &kub, &anorm, &bnorm, &modea, &modeb, &cndnma, &cndnmb, &dista, &distb)

					//                 Generate M by N matrix A
					matgen.Dlatms(&m, &n, dista, iseed, _type, rwork, &modea, &cndnma, &anorm, &kla, &kua, 'N', a, &lda, work, &iinfo)
					if iinfo != 0 {
						t.Fail()
						fmt.Printf(" DLATMS in DCKGQR:    INFO = %5d\n", iinfo)
						(*info) = abs(iinfo)
						goto label30
					}

					//                 Generate P by N matrix B
					matgen.Dlatms(&p, &n, distb, iseed, _type, rwork, &modeb, &cndnmb, &bnorm, &klb, &kub, 'N', b, &ldb, work, &iinfo)
					if iinfo != 0 {
						t.Fail()
						fmt.Printf(" DLATMS in DCKGQR:    INFO = %5d\n", iinfo)
						(*info) = abs(iinfo)
						goto label30
					}

					nt = 4

					Dgrqts(&m, &p, &n, a, af, aq, ar, &lda, taua, b, bf, bz, bt, bwk, &ldb, taub, work, &lwork, rwork, result)

					//                 Print information about the tests that did not
					//                 pass the threshold.
					for i = 1; i <= nt; i++ {
						if result.Get(i-1) >= (*thresh) {
							t.Fail()
							if nfail == 0 && firstt {
								firstt = false
								Alahdg([]byte("GRQ"))
							}
							fmt.Printf(" M=%4d P=%4d, N=%4d, _type %2d, test %2d, ratio=%13.6f\n", m, p, n, imat, i, result.Get(i-1))
							nfail = nfail + 1
						}
					}
					nrun = nrun + nt

					//                 Test DGGQRF
					//
					//                 Set up parameters with DLATB9 and generate test
					//                 matrices A and B with DLATMS.
					Dlatb9([]byte("GQR"), &imat, &m, &p, &n, &_type, &kla, &kua, &klb, &kub, &anorm, &bnorm, &modea, &modeb, &cndnma, &cndnmb, &dista, &distb)

					//                 Generate N-by-M matrix  A
					matgen.Dlatms(&n, &m, dista, iseed, _type, rwork, &modea, &cndnma, &anorm, &kla, &kua, 'N', a, &lda, work, &iinfo)
					if iinfo != 0 {
						t.Fail()
						fmt.Printf(" DLATMS in DCKGQR:    INFO = %5d\n", iinfo)
						(*info) = abs(iinfo)
						goto label30
					}

					//                 Generate N-by-P matrix  B
					matgen.Dlatms(&n, &p, distb, iseed, _type, rwork, &modea, &cndnma, &bnorm, &klb, &kub, 'N', b, &ldb, work, &iinfo)
					if iinfo != 0 {
						t.Fail()
						fmt.Printf(" DLATMS in DCKGQR:    INFO = %5d\n", iinfo)
						(*info) = abs(iinfo)
						goto label30
					}

					nt = 4

					Dgqrts(&n, &m, &p, a, af, aq, ar, &lda, taua, b, bf, bz, bt, bwk, &ldb, taub, work, &lwork, rwork, result)

					//                 Print information about the tests that did not
					//                 pass the threshold.
					for i = 1; i <= nt; i++ {
						if result.Get(i-1) >= (*thresh) {
							t.Fail()
							if nfail == 0 && firstt {
								firstt = false
								Alahdg(path)
							}
							fmt.Printf(" N=%4d M=%4d, P=%4d, _type %2d, test %2d, ratio=%13.6f\n", n, m, p, imat, i, result.Get(i-1))
							nfail = nfail + 1
						}
					}
					nrun = nrun + nt

				label30:
				}
			}
		}
	}

	//     Print a summary of the results.
	Alasum(path, &nfail, &nrun, func() *int { y := 0; return &y }())
}
