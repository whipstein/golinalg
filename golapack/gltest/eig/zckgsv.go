package eig

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// zckgsv tests Zggsvd:
//        the GSVD for M-by-N matrix A and P-by-N matrix B.
func zckgsv(nm int, mval, pval, nval []int, nmats int, iseed []int, thresh float64, nmax int, a, af, b, bf, u, v, q *mat.CVector, alpha, beta *mat.Vector, r *mat.CVector, iwork []int, work *mat.CVector, rwork *mat.Vector) (nfail, nrun int, err error) {
	var firstt bool
	var dista, distb, _type byte
	var anorm, bnorm, cndnma, cndnmb float64
	var i, iinfo, im, imat, kla, klb, kua, kub, lda, ldb, ldq, ldr, ldu, ldv, lwork, m, modea, modeb, n, nt, ntypes, p int
	dotype := make([]bool, 8)
	result := vf(12)

	// ntests = 12
	ntypes = 8

	//     Initialize constants and the random number seed.
	path := "Gsv"
	nrun = 0
	nfail = 0
	firstt = true
	alareq(nmats, &dotype)
	lda = nmax
	ldb = nmax
	ldu = nmax
	ldv = nmax
	ldq = nmax
	ldr = nmax
	lwork = nmax * nmax

	//     Do for each value of M in MVAL.
	for im = 1; im <= nm; im++ {
		m = mval[im-1]
		p = pval[im-1]
		n = nval[im-1]

		for imat = 1; imat <= ntypes; imat++ {
			//           Do the tests only if DOTYPE( IMAT ) is true.
			if !dotype[imat-1] {
				goto label20
			}

			//           Set up parameters with DLATB9 and generate test
			//           matrices A and B with Zlatms.
			_type, kla, kua, klb, kub, anorm, bnorm, modea, modeb, cndnma, cndnmb, dista, distb = dlatb9(path, imat, m, p, n)

			//           Generate M by N matrix A
			if err = matgen.Zlatms(m, n, dista, &iseed, _type, rwork, modea, cndnma, anorm, kla, kua, 'N', a.CMatrix(lda, opts), work); err != nil {
				fmt.Printf(" Zlatms in zckgsv   info = %5d\n", iinfo)
				err = fmt.Errorf("iinfo=%v", abs(iinfo))
				goto label20
			}

			//           Generate P by N matrix B
			if err = matgen.Zlatms(p, n, distb, &iseed, _type, rwork, modeb, cndnmb, bnorm, klb, kub, 'N', b.CMatrix(ldb, opts), work); err != nil {
				fmt.Printf(" Zlatms in zckgsv   info = %5d\n", iinfo)
				err = fmt.Errorf("iinfo=%v", abs(iinfo))
				goto label20
			}

			nt = 6

			zgsvts3(m, p, n, a.CMatrix(lda, opts), af.CMatrix(lda, opts), b.CMatrix(ldb, opts), bf.CMatrix(ldb, opts), u.CMatrix(ldu, opts), v.CMatrix(ldv, opts), q.CMatrix(ldq, opts), alpha, beta, r.CMatrix(ldr, opts), &iwork, work, lwork, rwork, result)

			//           Print information about the tests that did not
			//           pass the threshold.
			for i = 1; i <= nt; i++ {
				if result.Get(i-1) >= thresh {
					if nfail == 0 && firstt {
						firstt = false
						alahdg(path)
					}
					fmt.Printf(" m=%4d p=%4d, n=%4d, _type %2d, test %2d, ratio=%13.6f\n", m, p, n, imat, i, result.Get(i-1))
					err = fmt.Errorf(" m=%4d p=%4d, n=%4d, _type %2d, test %2d, ratio=%13.6f\n", m, p, n, imat, i, result.Get(i-1))
					nfail++
				}
			}
			nrun = nrun + nt

		label20:
		}
	}

	//     Print a summary of the results.
	// alasum(path, nfail, nrun, 0)

	return
}
