package eig

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// Dcklse tests DGGLSE - a subroutine for solving linear equality
// constrained least square problem (LSE).
func Dcklse(nn *int, mval, pval, nval *[]int, nmats *int, iseed *[]int, thresh *float64, nmax *int, a, af, b, bf, x, work, rwork *mat.Vector, nout, info *int, t *testing.T) {
	var firstt bool
	var dista, distb, _type byte
	var anorm, bnorm, cndnma, cndnmb float64
	var i, iinfo, ik, imat, kla, klb, kua, kub, lda, ldb, lwork, m, modea, modeb, n, nfail, nrun, nt, ntypes, p int

	dotype := make([]bool, 8)
	result := vf(7)

	ntypes = 8

	//     Initialize constants and the random number seed.
	path := []byte("LSE")
	(*info) = 0
	nrun = 0
	nfail = 0
	firstt = true
	Alareq(nmats, &dotype)
	lda = (*nmax)
	ldb = (*nmax)
	lwork = (*nmax) * (*nmax)

	//     Check for valid input values.
	for ik = 1; ik <= (*nn); ik++ {
		m = (*mval)[ik-1]
		p = (*pval)[ik-1]
		n = (*nval)[ik-1]
		if p > n || n > m+p {
			t.Fail()
			if firstt {
				fmt.Printf("\n")
				firstt = false
			}
			fmt.Printf(" *** Invalid input  for LSE:  M = %6d, P = %6d, N = %6d;\n     must satisfy P <= N <= P+M  (this set of values will be skipped)\n", m, p, n)
		}
	}
	firstt = true

	//     Do for each value of M in MVAL.
	for ik = 1; ik <= (*nn); ik++ {
		m = (*mval)[ik-1]
		p = (*pval)[ik-1]
		n = (*nval)[ik-1]
		if p > n || n > m+p {
			goto label40
		}

		for imat = 1; imat <= ntypes; imat++ {
			//           Do the tests only if DOTYPE( IMAT ) is true.
			if !dotype[imat-1] {
				goto label30
			}

			//           Set up parameters with DLATB9 and generate test
			//           matrices A and B with DLATMS.
			Dlatb9(path, &imat, &m, &p, &n, &_type, &kla, &kua, &klb, &kub, &anorm, &bnorm, &modea, &modeb, &cndnma, &cndnmb, &dista, &distb)

			matgen.Dlatms(&m, &n, dista, iseed, _type, rwork, &modea, &cndnma, &anorm, &kla, &kua, 'N', a.Matrix(lda, opts), &lda, work, &iinfo)
			if iinfo != 0 {
				t.Fail()
				fmt.Printf(" DLATMS in DCKLSE   INFO = %5d\n", iinfo)
				(*info) = abs(iinfo)
				goto label30
			}

			matgen.Dlatms(&p, &n, distb, iseed, _type, rwork, &modeb, &cndnmb, &bnorm, &klb, &kub, 'N', b.Matrix(ldb, opts), &ldb, work, &iinfo)
			if iinfo != 0 {
				t.Fail()
				fmt.Printf(" DLATMS in DCKLSE   INFO = %5d\n", iinfo)
				(*info) = abs(iinfo)
				goto label30
			}

			//           Generate the right-hand sides C and D for the LSE.
			Dlarhs([]byte("DGE"), toPtrByte('N'), 'U', 'N', &m, &n, toPtr(max(m-1, 0)), toPtr(max(n-1, 0)), func() *int { y := 1; return &y }(), a.Matrix(lda, opts), &lda, x.MatrixOff(4*(*nmax), max(n, 1), opts), toPtr(max(n, 1)), x.Matrix(max(m, 1), opts), toPtr(max(m, 1)), iseed, &iinfo)

			Dlarhs([]byte("DGE"), toPtrByte('C'), 'U', 'N', &p, &n, toPtr(max(p-1, 0)), toPtr(max(n-1, 0)), func() *int { y := 1; return &y }(), b.Matrix(ldb, opts), &ldb, x.MatrixOff(4*(*nmax), max(n, 1), opts), toPtr(max(n, 1)), x.MatrixOff(2*(*nmax), max(p, 1), opts), toPtr(max(p, 1)), iseed, &iinfo)

			nt = 2

			Dlsets(&m, &p, &n, a.Matrix(lda, opts), af.Matrix(lda, opts), &lda, b.Matrix(ldb, opts), bf.Matrix(ldb, opts), &ldb, x, x.Off((*nmax)), x.Off(2*(*nmax)), x.Off(3*(*nmax)), x.Off(4*(*nmax)), work, &lwork, rwork, result)

			//           Print information about the tests that did not
			//           pass the threshold.
			for i = 1; i <= nt; i++ {
				if result.Get(i-1) >= (*thresh) {
					t.Fail()
					if nfail == 0 && firstt {
						firstt = false
						Alahdg(path)
					}
					fmt.Printf(" M=%4d P=%4d, N=%4d, _type %2d, test %2d, ratio=%13.6f\n", m, p, n, imat, i, result.Get(i-1))
					nfail = nfail + 1
				}
			}
			nrun = nrun + nt

		label30:
		}
	label40:
	}

	//     Print a summary of the results.
	Alasum(path, &nfail, &nrun, func() *int { y := 0; return &y }())
}
