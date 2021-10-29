package eig

import (
	"fmt"
	"math"
	"testing"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// dckcsd tests DORCSD:
//        the CSD for an M-by-M orthogonal matrix X partitioned as
//        [ X11 X12; X21 X22 ]. X11 is P-by-Q.
func dckcsd(nm int, mval []int, pval []int, qval []int, nmats int, iseed []int, thresh float64, mmax int, x, xf, u1, u2, v1t, v2t, theta *mat.Vector, iwork []int, work, rwork *mat.Vector, nout int, t *testing.T) (err error) {
	var firstt bool
	var gapdigit, one, orth, piover2, ten, zero float64
	var i, iinfo, im, imat, j, ldu1, ldu2, ldv1t, ldv2t, ldx, lwork, m, nfail, nrun, nt, ntypes, p, q, r int

	dotype := make([]bool, 4)
	result := vf(15)

	ntypes = 4
	gapdigit = 18.0
	one = 1.0
	orth = 1.0e-12
	piover2 = 1.57079632679489662
	ten = 10.0
	zero = 0.0

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
				err = matgen.Dlaror('L', 'I', m, m, x.Matrix(ldx, opts), &iseed, work)
				if m != 0 && err != nil {
					t.Fail()
					fmt.Printf(" Dlaror in dchkcsd: M = %5d, info = %15d\n", m, iinfo)
					err = fmt.Errorf("iinfo=%v", abs(iinfo))
					goto label20
				}
			} else if imat == 2 {
				r = min(p, m-p, q, m-q)
				for i = 1; i <= r; i++ {
					theta.Set(i-1, piover2*matgen.Dlarnd(1, &iseed))
				}
				dlacsg(&m, &p, &q, theta, &iseed, x.Matrix(ldx, opts), &ldx, work)
				for i = 1; i <= m; i++ {
					for j = 1; j <= m; j++ {
						x.Set(i+(j-1)*ldx-1, x.Get(i+(j-1)*ldx-1)+orth*matgen.Dlarnd(2, &iseed))
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
				dlacsg(&m, &p, &q, theta, &iseed, x.Matrix(ldx, opts), &ldx, work)
			} else {
				golapack.Dlaset(Full, m, m, zero, one, x.Matrix(ldx, opts))
				for i = 1; i <= m; i++ {
					j = int(matgen.Dlaran(&iseed))*m + 1
					if j != i {
						goblas.Drot(m, x.Off(1+(i-1)*ldx-1, 1), x.Off(1+(j-1)*ldx-1, 1), zero, one)
					}
				}
			}

			nt = 15

			dcsdts(m, p, q, x.Matrix(ldx, opts), xf.Matrix(ldx, opts), u1.Matrix(ldu1, opts), u2.Matrix(ldu2, opts), v1t.Matrix(ldv1t, opts), v2t.Matrix(ldv2t, opts), theta, iwork, work, lwork, rwork, result)

			//           Print information about the tests that did not
			//           pass the threshold.
			for i = 1; i <= nt; i++ {
				if result.Get(i-1) >= thresh {
					t.Fail()
					if nfail == 0 && firstt {
						firstt = false
						alahdg(path)
					}
					fmt.Printf(" M=%4d P=%4d, Q=%4d, type %2d, test %2d, ratio=%13.6f\n", m, p, q, imat, i, result.Get(i-1))
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

func dlacsg(m, p, q *int, theta *mat.Vector, iseed *[]int, x *mat.Matrix, ldx *int, work *mat.Vector) {
	var one, zero float64
	var i, r int
	var err error

	one = 1.0
	zero = 0.0

	r = min(*p, (*m)-(*p), *q, (*m)-(*q))

	golapack.Dlaset(Full, *m, *m, zero, zero, x)

	for i = 1; i <= min(*p, *q)-r; i++ {
		x.Set(i-1, i-1, one)
	}
	for i = 1; i <= r; i++ {
		x.Set(min(*p, *q)-r+i-1, min(*p, *q)-r+i-1, math.Cos(theta.Get(i-1)))
	}
	for i = 1; i <= min(*p, (*m)-(*q))-r; i++ {
		x.Set((*p)-i, (*m)-i, -one)
	}
	for i = 1; i <= r; i++ {
		x.Set((*p)-(min(*p, (*m)-(*q))-r)+1-i-1, (*m)-(min(*p, (*m)-(*q))-r)+1-i-1, -math.Sin(theta.Get(r-i)))
	}
	for i = 1; i <= min((*m)-(*p), *q)-r; i++ {
		x.Set((*m)-i, (*q)-i, one)
	}
	for i = 1; i <= r; i++ {
		x.Set((*m)-(min((*m)-(*p), *q)-r)+1-i-1, (*q)-(min((*m)-(*p), *q)-r)+1-i-1, math.Sin(theta.Get(r-i)))
	}
	for i = 1; i <= min((*m)-(*p), (*m)-(*q))-r; i++ {
		x.Set((*p)+i-1, (*q)+i-1, one)
	}
	for i = 1; i <= r; i++ {
		x.Set((*p)+(min((*m)-(*p), (*m)-(*q))-r)+i-1, (*q)+(min((*m)-(*p), (*m)-(*q))-r)+i-1, math.Cos(theta.Get(i-1)))
	}
	if err = matgen.Dlaror('L', 'N', *p, *m, x, iseed, work); err != nil {
		panic(err)
	}
	if err = matgen.Dlaror('L', 'N', (*m)-(*p), *m, x.Off(*p, 0), iseed, work); err != nil {
		panic(err)
	}
	if err = matgen.Dlaror('R', 'N', *m, *q, x, iseed, work); err != nil {
		panic(err)
	}
	if err = matgen.Dlaror('R', 'N', *m, (*m)-(*q), x.Off(0, *q), iseed, work); err != nil {
		panic(err)
	}
}
