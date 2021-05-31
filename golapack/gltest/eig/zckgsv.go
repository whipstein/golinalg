package eig

import (
	"fmt"
	"testing"

	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// Zckgsv tests ZGGSVD:
//        the GSVD for M-by-N matrix A and P-by-N matrix B.
func Zckgsv(nm *int, mval, pval, nval *[]int, nmats *int, iseed *[]int, thresh *float64, nmax *int, a, af, b, bf, u, v, q *mat.CVector, alpha, beta *mat.Vector, r *mat.CVector, iwork *[]int, work *mat.CVector, rwork *mat.Vector, nout, info *int, t *testing.T) {
	var firstt bool
	var dista, distb, _type byte
	var anorm, bnorm, cndnma, cndnmb float64
	var i, iinfo, im, imat, kla, klb, kua, kub, lda, ldb, ldq, ldr, ldu, ldv, lwork, m, modea, modeb, n, nfail, nrun, nt, ntypes, p int
	dotype := make([]bool, 8)
	result := vf(12)

	// ntests = 12
	ntypes = 8

	//     Initialize constants and the random number seed.
	path := []byte("GSV")
	(*info) = 0
	nrun = 0
	nfail = 0
	firstt = true
	Alareq(nmats, &dotype)
	lda = (*nmax)
	ldb = (*nmax)
	ldu = (*nmax)
	ldv = (*nmax)
	ldq = (*nmax)
	ldr = (*nmax)
	lwork = (*nmax) * (*nmax)

	//     Do for each value of M in MVAL.
	for im = 1; im <= (*nm); im++ {
		m = (*mval)[im-1]
		p = (*pval)[im-1]
		n = (*nval)[im-1]

		for imat = 1; imat <= ntypes; imat++ {
			//           Do the tests only if DOTYPE( IMAT ) is true.
			if !dotype[imat-1] {
				goto label20
			}

			//           Set up parameters with DLATB9 and generate test
			//           matrices A and B with ZLATMS.
			Dlatb9(path, &imat, &m, &p, &n, &_type, &kla, &kua, &klb, &kub, &anorm, &bnorm, &modea, &modeb, &cndnma, &cndnmb, &dista, &distb)

			//           Generate M by N matrix A
			matgen.Zlatms(&m, &n, dista, iseed, _type, rwork, &modea, &cndnma, &anorm, &kla, &kua, 'N', a.CMatrix(lda, opts), &lda, work, &iinfo)
			if iinfo != 0 {
				t.Fail()
				fmt.Printf(" ZLATMS in ZCKGSV   INFO = %5d\n", iinfo)
				(*info) = absint(iinfo)
				goto label20
			}

			//           Generate P by N matrix B
			matgen.Zlatms(&p, &n, distb, iseed, _type, rwork, &modeb, &cndnmb, &bnorm, &klb, &kub, 'N', b.CMatrix(ldb, opts), &ldb, work, &iinfo)
			if iinfo != 0 {
				t.Fail()
				fmt.Printf(" ZLATMS in ZCKGSV   INFO = %5d\n", iinfo)
				(*info) = absint(iinfo)
				goto label20
			}

			nt = 6

			Zgsvts3(&m, &p, &n, a.CMatrix(lda, opts), af.CMatrix(lda, opts), &lda, b.CMatrix(ldb, opts), bf.CMatrix(ldb, opts), &ldb, u.CMatrix(ldu, opts), &ldu, v.CMatrix(ldv, opts), &ldv, q.CMatrix(ldq, opts), &ldq, alpha, beta, r.CMatrix(ldr, opts), &ldr, iwork, work, &lwork, rwork, result)

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

		label20:
		}
	}

	//     Print a summary of the results.
	Alasum(path, &nfail, &nrun, func() *int { y := 0; return &y }())
}
