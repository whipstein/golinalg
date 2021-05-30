package eig

import (
	"fmt"
	"golinalg/goblas"
	"golinalg/golapack"
	"golinalg/golapack/gltest/matgen"
	"golinalg/mat"
	"math"
	"testing"
)

// Zckcsd tests ZUNCSD:
//        the CSD for an M-by-M unitary matrix X partitioned as
//        [ X11 X12; X21 X22 ]. X11 is P-by-Q.
func Zckcsd(nm *int, mval *[]int, pval *[]int, qval *[]int, nmats *int, iseed *[]int, thresh *float64, mmax *int, x, xf, u1, u2, v1t, v2t *mat.CVector, theta *mat.Vector, iwork *[]int, work *mat.CVector, rwork *mat.Vector, nout, info *int, t *testing.T) {
	var firstt bool
	var one, zero complex128
	var gapdigit, orth, piover2, realone, realzero, ten float64
	var i, iinfo, im, imat, j, ldu1, ldu2, ldv1t, ldv2t, ldx, lwork, m, nfail, nrun, nt, ntypes, p, q, r int
	dotype := make([]bool, 4)
	result := vf(15)

	// ntests = 15
	ntypes = 4
	gapdigit = 18.0
	orth = 1.0e-12
	piover2 = 1.57079632679489662
	realone = 1.0
	realzero = 0.0
	ten = 10.0
	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	//     Initialize constants and the random number seed.
	path := []byte("CSD")
	(*info) = 0
	nrun = 0
	nfail = 0
	firstt = true
	Alareq(nmats, &dotype)
	ldx = (*mmax)
	ldu1 = (*mmax)
	ldu2 = (*mmax)
	ldv1t = (*mmax)
	ldv2t = (*mmax)
	lwork = (*mmax) * (*mmax)

	//     Do for each value of M in MVAL.
	for im = 1; im <= (*nm); im++ {
		m = (*mval)[im-1]
		p = (*pval)[im-1]
		q = (*qval)[im-1]

		for imat = 1; imat <= ntypes; imat++ {
			//           Do the tests only if DOTYPE( IMAT ) is true.
			if !dotype[imat-1] {
				goto label20
			}

			//           Generate X
			if imat == 1 {
				matgen.Zlaror('L', 'I', &m, &m, x.CMatrix(ldx, opts), &ldx, iseed, work, &iinfo)
				if m != 0 && iinfo != 0 {
					t.Fail()
					fmt.Printf(" ZLAROR in ZCKCSD: M = %5d, INFO = %15d\n", m, iinfo)
					(*info) = absint(iinfo)
					goto label20
				}
			} else if imat == 2 {
				r = minint(p, m-p, q, m-q)
				for i = 1; i <= r; i++ {
					theta.Set(i-1, piover2*matgen.Dlarnd(func() *int { y := 1; return &y }(), iseed))
				}
				Zlacsg(&m, &p, &q, theta, iseed, x.CMatrix(ldx, opts), &ldx, work)
				for i = 1; i <= m; i++ {
					for j = 1; j <= m; j++ {
						x.Set(i+(j-1)*ldx-1, x.Get(i+(j-1)*ldx-1)+toCmplx(orth*matgen.Dlarnd(func() *int { y := 2; return &y }(), iseed)))
					}
				}
			} else if imat == 3 {
				r = minint(p, m-p, q, m-q)
				for i = 1; i <= r+1; i++ {
					theta.Set(i-1, math.Pow(ten, -matgen.Dlarnd(func() *int { y := 1; return &y }(), iseed)*gapdigit))
				}
				for i = 2; i <= r+1; i++ {
					theta.Set(i-1, theta.Get(i-1-1)+theta.Get(i-1))
				}
				for i = 1; i <= r; i++ {
					theta.Set(i-1, piover2*theta.Get(i-1)/theta.Get(r+1-1))
				}
				Zlacsg(&m, &p, &q, theta, iseed, x.CMatrix(ldx, opts), &ldx, work)
			} else {
				golapack.Zlaset('F', &m, &m, &zero, &one, x.CMatrix(ldx, opts), &ldx)
				for i = 1; i <= m; i++ {
					j = int(matgen.Dlaran(iseed)*float64(m)) + 1
					if j != i {
						goblas.Zdrot(&m, x.Off(1+(i-1)*ldx-1), func() *int { y := 1; return &y }(), x.Off(1+(j-1)*ldx-1), func() *int { y := 1; return &y }(), &realzero, &realone)
					}
				}
			}

			nt = 15

			Zcsdts(&m, &p, &q, x.CMatrix(ldx, opts), xf.CMatrix(ldx, opts), &ldx, u1.CMatrix(ldu1, opts), &ldu1, u2.CMatrix(ldu2, opts), &ldu2, v1t.CMatrix(ldv1t, opts), &ldv1t, v2t.CMatrix(ldv2t, opts), &ldv2t, theta, iwork, work, &lwork, rwork, result)

			//           Print information about the tests that did not
			//           pass the threshold.
			for i = 1; i <= nt; i++ {
				if result.Get(i-1) >= (*thresh) {
					t.Fail()
					if nfail == 0 && firstt {
						firstt = false
						Alahdg(path)
					}
					fmt.Printf(" M=%4d P=%4d, Q=%4d, _type %2d, test %2d, ratio=%13.6f\n", m, p, q, imat, i, result.Get(i-1))
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

func Zlacsg(m, p, q *int, theta *mat.Vector, iseed *[]int, x *mat.CMatrix, ldx *int, work *mat.CVector) {
	var one, zero complex128
	var i, info, r int

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	r = minint(*p, (*m)-(*p), *q, (*m)-(*q))

	golapack.Zlaset('F', m, m, &zero, &zero, x, ldx)

	for i = 1; i <= minint(*p, *q)-r; i++ {
		x.Set(i-1, i-1, one)
	}
	for i = 1; i <= r; i++ {
		x.Set(minint(*p, *q)-r+i-1, minint(*p, *q)-r+i-1, toCmplx(math.Cos(theta.Get(i-1))))
	}
	for i = 1; i <= minint(*p, (*m)-(*q))-r; i++ {
		x.Set((*p)-i+1-1, (*m)-i+1-1, -one)
	}
	for i = 1; i <= r; i++ {
		x.Set((*p)-(minint(*p, (*m)-(*q))-r)+1-i-1, (*m)-(minint(*p, (*m)-(*q))-r)+1-i-1, toCmplx(-math.Sin(theta.Get(r-i+1-1))))
	}
	for i = 1; i <= minint((*m)-(*p), *q)-r; i++ {
		x.Set((*m)-i+1-1, (*q)-i+1-1, one)
	}
	for i = 1; i <= r; i++ {
		x.Set((*m)-(minint((*m)-(*p), *q)-r)+1-i-1, (*q)-(minint((*m)-(*p), *q)-r)+1-i-1, toCmplx(math.Sin(theta.Get(r-i+1-1))))
	}
	for i = 1; i <= minint((*m)-(*p), (*m)-(*q))-r; i++ {
		x.Set((*p)+i-1, (*q)+i-1, one)
	}
	for i = 1; i <= r; i++ {
		x.Set((*p)+(minint((*m)-(*p), (*m)-(*q))-r)+i-1, (*q)+(minint((*m)-(*p), (*m)-(*q))-r)+i-1, toCmplx(math.Cos(theta.Get(i-1))))
	}
	matgen.Zlaror('L', 'N', p, m, x, ldx, iseed, work, &info)
	matgen.Zlaror('L', 'N', toPtr((*m)-(*p)), m, x.Off((*p)+1-1, 0), ldx, iseed, work, &info)
	matgen.Zlaror('R', 'N', m, q, x, ldx, iseed, work, &info)
	matgen.Zlaror('R', 'N', m, toPtr((*m)-(*q)), x.Off(0, (*q)+1-1), ldx, iseed, work, &info)

}