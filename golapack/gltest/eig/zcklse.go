package eig

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// zcklse tests Zgglse - a subroutine for solving linear equality
// constrained least square problem (Lse).
func zcklse(nn int, mval, pval, nval []int, nmats int, iseed []int, thresh float64, nmax int, a, af, b, bf, x, work *mat.CVector, rwork *mat.Vector) (err error) {
	var firstt bool
	var dista, distb, _type byte
	var anorm, bnorm, cndnma, cndnmb float64
	var i, iinfo, ik, imat, kla, klb, kua, kub, lda, ldb, lwork, m, modea, modeb, n, nfail, nrun, nt, ntypes, p int
	dotype := make([]bool, 8)
	result := vf(7)

	// ntests = 7
	ntypes = 8

	//     Initialize constants and the random number seed.
	path := "Lse"
	nrun = 0
	nfail = 0
	firstt = true
	alareq(nmats, &dotype)
	lda = nmax
	ldb = nmax
	lwork = nmax * nmax

	//     Check for valid input values.
	for ik = 1; ik <= nn; ik++ {
		m = mval[ik-1]
		p = pval[ik-1]
		n = nval[ik-1]
		if p > n || n > m+p {
			if firstt {
				fmt.Printf("\n")
				firstt = false
			}
			fmt.Printf(" *** Invalid input  for Lse:  m = %6d, p = %6d, n = %6d;\n     must satisfy p <= n <= p+m  (this set of values will be skipped)\n", m, p, n)
		}
	}
	firstt = true

	//     Do for each value of M in MVAL.
	for ik = 1; ik <= nn; ik++ {
		m = mval[ik-1]
		p = pval[ik-1]
		n = nval[ik-1]
		if p > n || n > m+p {
			goto label40
		}

		for imat = 1; imat <= ntypes; imat++ {
			//           Do the tests only if DOTYPE( IMAT ) is true.
			if !dotype[imat-1] {
				goto label30
			}

			//           Set up parameters with DLATB9 and generate test
			//           matrices A and B with Zlatms.
			_type, kla, kua, klb, kub, anorm, bnorm, modea, modeb, cndnma, cndnmb, dista, distb = dlatb9(path, imat, m, p, n)

			if err = matgen.Zlatms(m, n, dista, &iseed, _type, rwork, modea, cndnma, anorm, kla, kua, 'N', a.CMatrix(lda, opts), work); err != nil {
				fmt.Printf(" Zlatms in zcklse   info = %5d\n", iinfo)
				err = fmt.Errorf("iinfo=%v", abs(iinfo))
				goto label30
			}

			if err = matgen.Zlatms(p, n, distb, &iseed, _type, rwork, modeb, cndnmb, bnorm, klb, kub, 'N', b.CMatrix(ldb, opts), work); err != nil {
				fmt.Printf(" Zlatms in zcklse   info = %5d\n", iinfo)
				err = fmt.Errorf("iinfo=%v", abs(iinfo))
				goto label30
			}

			//           Generate the right-hand sides C and D for the Lse.
			if err = zlarhs("Zge", 'N', Upper, NoTrans, m, n, max(m-1, 0), max(n-1, 0), 1, a.CMatrix(lda, opts), x.CMatrixOff(4*nmax, max(n, 1), opts), x.CMatrix(max(m, 1), opts), &iseed); err != nil {
				panic(err)
			}

			if err = zlarhs("Zge", 'C', Upper, NoTrans, p, n, max(p-1, 0), max(n-1, 0), 1, b.CMatrix(ldb, opts), x.CMatrixOff(4*nmax, max(n, 1), opts), x.CMatrixOff(2*nmax, max(p, 1), opts), &iseed); err != nil {
				panic(err)
			}

			nt = 2

			zlsets(m, p, n, a.CMatrix(lda, opts), af.CMatrix(lda, opts), b.CMatrix(ldb, opts), bf.CMatrix(ldb, opts), x, x.Off(nmax), x.Off(2*nmax), x.Off(3*nmax), x.Off(4*nmax), work, lwork, rwork, result.Off(0))

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

		label30:
		}
	label40:
	}

	//     Print a summary of the results.
	alasum(path, nfail, nrun, 0)

	return
}
