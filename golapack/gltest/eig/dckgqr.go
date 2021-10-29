package eig

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// dckgqr tests
// DGGQRF: GQR factorization for N-by-M matrix A and N-by-P matrix B,
// DGGRQF: GRQ factorization for M-by-N matrix A and P-by-N matrix B.
func dckgqr(nm int, mval []int, np int, pval []int, nn int, nval []int, nmats int, iseed []int, thresh float64, nmax int, a, af, aq, ar *mat.Matrix, taua *mat.Vector, b, bf, bz, bt, bwk *mat.Matrix, taub, work, rwork *mat.Vector, nout int, t *testing.T) (err error) {
	var firstt bool
	var dista, distb, _type byte
	var anorm, bnorm, cndnma, cndnmb float64
	var i, iinfo, im, imat, in, ip, kla, klb, kua, kub, lwork, m, modea, modeb, n, nfail, nrun, nt, ntypes, p int

	dotype := make([]bool, 8)
	result := vf(7)

	ntypes = 8

	//     Initialize constants.
	path := "Gqr"
	nrun = 0
	nfail = 0
	firstt = true
	alareq(nmats, &dotype)
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

					//                 Test DGGRQF
					//
					//                 Set up parameters with DLATB9 and generate test
					//                 matrices A and B with DLATMS.
					_type, kla, kua, klb, kub, anorm, bnorm, modea, modeb, cndnma, cndnmb, dista, distb = dlatb9("Grq", imat, m, p, n)

					//                 Generate M by N matrix A
					if iinfo, err = matgen.Dlatms(m, n, dista, &iseed, _type, rwork, modea, cndnma, anorm, kla, kua, 'N', a, work); iinfo != 0 {
						t.Fail()
						fmt.Printf(" DLATMS in DCKGQR:    INFO = %5d\n", iinfo)
						err = fmt.Errorf("iinfo=%v", abs(iinfo))
						goto label30
					}

					//                 Generate P by N matrix B
					if iinfo, err = matgen.Dlatms(p, n, distb, &iseed, _type, rwork, modeb, cndnmb, bnorm, klb, kub, 'N', b, work); iinfo != 0 {
						t.Fail()
						fmt.Printf(" DLATMS in DCKGQR:    INFO = %5d\n", iinfo)
						err = fmt.Errorf("iinfo=%v", abs(iinfo))
						goto label30
					}

					nt = 4

					dgrqts(m, p, n, a, af, aq, ar, taua, b, bf, bz, bt, bwk, taub, work, lwork, rwork, result)

					//                 Print information about the tests that did not
					//                 pass the threshold.
					for i = 1; i <= nt; i++ {
						if result.Get(i-1) >= thresh {
							t.Fail()
							if nfail == 0 && firstt {
								firstt = false
								alahdg("Grq")
							}
							fmt.Printf(" M=%4d P=%4d, N=%4d, _type %2d, test %2d, ratio=%13.6f\n", m, p, n, imat, i, result.Get(i-1))
							nfail++
						}
					}
					nrun = nrun + nt

					//                 Test DGGQRF
					//
					//                 Set up parameters with DLATB9 and generate test
					//                 matrices A and B with DLATMS.
					_type, kla, kua, klb, kub, anorm, bnorm, modea, modeb, cndnma, cndnmb, dista, distb = dlatb9("Gqr", imat, m, p, n)

					//                 Generate N-by-M matrix  A
					if iinfo, _ = matgen.Dlatms(n, m, dista, &iseed, _type, rwork, modea, cndnma, anorm, kla, kua, 'N', a, work); iinfo != 0 {
						t.Fail()
						fmt.Printf(" DLATMS in DCKGQR:    INFO = %5d\n", iinfo)
						err = fmt.Errorf("iinfo=%v", abs(iinfo))
						goto label30
					}

					//                 Generate N-by-P matrix  B
					if iinfo, _ = matgen.Dlatms(n, p, distb, &iseed, _type, rwork, modea, cndnma, bnorm, klb, kub, 'N', b, work); iinfo != 0 {
						t.Fail()
						fmt.Printf(" DLATMS in DCKGQR:    INFO = %5d\n", iinfo)
						err = fmt.Errorf("iinfo=%v", abs(iinfo))
						goto label30
					}

					nt = 4

					dgqrts(n, m, p, a, af, aq, ar, taua, b, bf, bz, bt, bwk, taub, work, lwork, rwork, result)

					//                 Print information about the tests that did not
					//                 pass the threshold.
					for i = 1; i <= nt; i++ {
						if result.Get(i-1) >= thresh {
							t.Fail()
							if nfail == 0 && firstt {
								firstt = false
								alahdg(path)
							}
							fmt.Printf(" N=%4d M=%4d, P=%4d, _type %2d, test %2d, ratio=%13.6f\n", n, m, p, imat, i, result.Get(i-1))
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
	alasum(path, nfail, nrun, 0)

	return
}
