package eig

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// zckcsd tests ZUNCSD:
//        the CSD for an M-by-M unitary matrix X partitioned as
//        [ X11 X12; X21 X22 ]. X11 is P-by-Q.
func zckcsd(nm int, mval []int, pval []int, qval []int, nmats int, iseed []int, thresh float64, mmax int, x, xf, u1, u2, v1t, v2t *mat.CVector, theta *mat.Vector, iwork []int, work *mat.CVector, rwork *mat.Vector) (err error) {
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
	path := "Csd"
	nrun = 0
	nfail = 0
	firstt = true
	alareq(nmats, &dotype)
	ldx = mmax
	ldu1 = mmax
	ldu2 = mmax
	ldv1t = mmax
	ldv2t = mmax
	lwork = mmax * mmax

	//     Do for each value of M in MVAL.
	for im = 1; im <= nm; im++ {
		m = mval[im-1]
		p = pval[im-1]
		q = qval[im-1]

		for imat = 1; imat <= ntypes; imat++ {
			//           Do the tests only if DOTYPE( IMAT ) is true.
			if !dotype[imat-1] {
				goto label20
			}

			//           Generate X
			if imat == 1 {
				if err = matgen.Zlaror('L', 'I', m, m, x.CMatrix(ldx, opts), &iseed, work); m != 0 && err != nil {
					fmt.Printf(" Zlaror in zckcsd: m = %5d, info = %15d\n", m, iinfo)
					err = fmt.Errorf("iinfo=%v", abs(iinfo))
					goto label20
				}
			} else if imat == 2 {
				r = min(p, m-p, q, m-q)
				for i = 1; i <= r; i++ {
					theta.Set(i-1, piover2*matgen.Dlarnd(1, &iseed))
				}
				err = Zlacsg(m, p, q, theta, &iseed, x.CMatrix(ldx, opts), work)
				for i = 1; i <= m; i++ {
					for j = 1; j <= m; j++ {
						x.Set(i+(j-1)*ldx-1, x.Get(i+(j-1)*ldx-1)+toCmplx(orth*matgen.Dlarnd(2, &iseed)))
					}
				}
			} else if imat == 3 {
				r = min(p, m-p, q, m-q)
				for i = 1; i <= r+1; i++ {
					theta.Set(i-1, math.Pow(ten, -matgen.Dlarnd(1, &iseed)*gapdigit))
				}
				for i = 2; i <= r+1; i++ {
					theta.Set(i-1, theta.Get(i-1-1)+theta.Get(i-1))
				}
				for i = 1; i <= r; i++ {
					theta.Set(i-1, piover2*theta.Get(i-1)/theta.Get(r))
				}
				err = Zlacsg(m, p, q, theta, &iseed, x.CMatrix(ldx, opts), work)
			} else {
				golapack.Zlaset(Full, m, m, zero, one, x.CMatrix(ldx, opts))
				for i = 1; i <= m; i++ {
					j = int(matgen.Dlaran(&iseed)*float64(m)) + 1
					if j != i {
						goblas.Zdrot(m, x.Off(1+(i-1)*ldx-1, 1), x.Off(1+(j-1)*ldx-1, 1), realzero, realone)
					}
				}
			}

			nt = 15

			zcsdts(m, p, q, x.CMatrix(ldx, opts), xf.CMatrix(ldx, opts), u1.CMatrix(ldu1, opts), u2.CMatrix(ldu2, opts), v1t.CMatrix(ldv1t, opts), v2t.CMatrix(ldv2t, opts), theta, iwork, work, lwork, rwork, result)

			//           Print information about the tests that did not
			//           pass the threshold.
			for i = 1; i <= nt; i++ {
				if result.Get(i-1) >= thresh {
					if nfail == 0 && firstt {
						firstt = false
						alahdg(path)
					}
					fmt.Printf(" m=%4d p=%4d, q=%4d, _type %2d, test %2d, ratio=%13.6f\n", m, p, q, imat, i, result.Get(i-1))
					err = fmt.Errorf(" m=%4d p=%4d, q=%4d, _type %2d, test %2d, ratio=%13.6f\n", m, p, q, imat, i, result.Get(i-1))
					nfail++
				}
			}
			nrun = nrun + nt
		label20:
		}
	}

	//     Print a summary of the results.
	alasum(path, nfail, nrun, 0)

	return
}

func Zlacsg(m, p, q int, theta *mat.Vector, iseed *[]int, x *mat.CMatrix, work *mat.CVector) (err error) {
	var one, zero complex128
	var i, r int

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	r = min(p, m-p, q, m-q)

	golapack.Zlaset(Full, m, m, zero, zero, x)

	for i = 1; i <= min(p, q)-r; i++ {
		x.Set(i-1, i-1, one)
	}
	for i = 1; i <= r; i++ {
		x.Set(min(p, q)-r+i-1, min(p, q)-r+i-1, toCmplx(math.Cos(theta.Get(i-1))))
	}
	for i = 1; i <= min(p, m-q)-r; i++ {
		x.Set(p-i, m-i, -one)
	}
	for i = 1; i <= r; i++ {
		x.Set(p-(min(p, m-q)-r)+1-i-1, m-(min(p, m-q)-r)+1-i-1, toCmplx(-math.Sin(theta.Get(r-i))))
	}
	for i = 1; i <= min(m-p, q)-r; i++ {
		x.Set(m-i, q-i, one)
	}
	for i = 1; i <= r; i++ {
		x.Set(m-(min(m-p, q)-r)+1-i-1, q-(min(m-p, q)-r)+1-i-1, toCmplx(math.Sin(theta.Get(r-i))))
	}
	for i = 1; i <= min(m-p, m-q)-r; i++ {
		x.Set(p+i-1, q+i-1, one)
	}
	for i = 1; i <= r; i++ {
		x.Set(p+(min(m-p, m-q)-r)+i-1, q+(min(m-p, m-q)-r)+i-1, toCmplx(math.Cos(theta.Get(i-1))))
	}
	err = matgen.Zlaror('L', 'N', p, m, x, iseed, work)
	err = matgen.Zlaror('L', 'N', m-p, m, x.Off(p, 0), iseed, work)
	err = matgen.Zlaror('R', 'N', m, q, x, iseed, work)
	err = matgen.Zlaror('R', 'N', m, m-q, x.Off(0, q), iseed, work)

	return
}
