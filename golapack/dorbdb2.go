package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dorbdb2 simultaneously bidiagonalizes the blocks of a tall and skinny
// matrix X with orthonomal columns:
//
//                            [ B11 ]
//      [ X11 ]   [ P1 |    ] [  0  ]
//      [-----] = [---------] [-----] Q1**T .
//      [ X21 ]   [    | P2 ] [ B21 ]
//                            [  0  ]
//
// X11 is P-by-Q, and X21 is (M-P)-by-Q. P must be no larger than M-P,
// Q, or M-Q. Routines DORBDB1, DORBDB3, and DORBDB4 handle cases in
// which P is not the minimum dimension.
//
// The orthogonal matrices P1, P2, and Q1 are P-by-P, (M-P)-by-(M-P),
// and (M-Q)-by-(M-Q), respectively. They are represented implicitly by
// Householder vectors.
//
// B11 and B12 are P-by-P bidiagonal matrices represented implicitly by
// angles THETA, PHI.
func Dorbdb2(m, p, q int, x11, x21 *mat.Matrix, theta, phi, taup1, taup2, tauq1, work *mat.Vector, lwork int) (err error) {
	var lquery bool
	var c, negone, one, s float64
	var i, ilarf, iorbdb5, llarf, lorbdb5, lworkmin, lworkopt int

	negone = -1.0
	one = 1.0

	//     Test input arguments
	lquery = lwork == -1

	if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if p < 0 || p > m-p {
		err = fmt.Errorf("p < 0 || p > m-p: p=%v, m=%v", p, m)
	} else if q < 0 || q < p || m-q < p {
		err = fmt.Errorf("q < 0 || q < p || m-q < p: q=%v, p=%v, m=%v", q, p, m)
	} else if x11.Rows < max(1, p) {
		err = fmt.Errorf("x11.Rows < max(1, p): x11.Rows=%v, p=%v", x11.Rows, p)
	} else if x21.Rows < max(1, m-p) {
		err = fmt.Errorf("x21.Rows < max(1, m-p): x21.Rows=%v, m=%v, p=%v", x21.Rows, m, p)
	}

	//     Compute workspace
	if err == nil {
		ilarf = 2
		llarf = max(p-1, m-p, q-1)
		iorbdb5 = 2
		lorbdb5 = q - 1
		lworkopt = max(ilarf+llarf-1, iorbdb5+lorbdb5-1)
		lworkmin = lworkopt
		work.Set(0, float64(lworkopt))
		if lwork < lworkmin && !lquery {
			err = fmt.Errorf("lwork < lworkmin && !lquery: lwork=%v, lworkmin=%v, lquery=%v", lwork, lworkmin, lquery)
		}
	}
	if err != nil {
		gltest.Xerbla2("Dorbdb2", err)
		return
	} else if lquery {
		return
	}

	//     Reduce rows 1, ..., P of X11 and X21
	for i = 1; i <= p; i++ {

		if i > 1 {
			goblas.Drot(q-i+1, x11.Vector(i-1, i-1), x21.Vector(i-1-1, i-1), c, s)
		}
		*x11.GetPtr(i-1, i-1), *tauq1.GetPtr(i - 1) = Dlarfgp(q-i+1, x11.Get(i-1, i-1), x11.Vector(i-1, i))
		c = x11.Get(i-1, i-1)
		x11.Set(i-1, i-1, one)
		Dlarf(Right, p-i, q-i+1, x11.Vector(i-1, i-1), tauq1.Get(i-1), x11.Off(i, i-1), work.Off(ilarf-1))
		Dlarf(Right, m-p-i+1, q-i+1, x11.Vector(i-1, i-1), tauq1.Get(i-1), x21.Off(i-1, i-1), work.Off(ilarf-1))
		s = math.Sqrt(math.Pow(goblas.Dnrm2(p-i, x11.Vector(i, i-1, 1)), 2) + math.Pow(goblas.Dnrm2(m-p-i+1, x21.Vector(i-1, i-1, 1)), 2))
		theta.Set(i-1, math.Atan2(s, c))

		if err = Dorbdb5(p-i, m-p-i+1, q-i, x11.Vector(i, i-1, 1), x21.Vector(i-1, i-1, 1), x11.Off(i, i), x21.Off(i-1, i), work.Off(iorbdb5-1), lorbdb5); err != nil {
			panic(err)
		}
		goblas.Dscal(p-i, negone, x11.Vector(i, i-1, 1))
		*x21.GetPtr(i-1, i-1), *taup2.GetPtr(i - 1) = Dlarfgp(m-p-i+1, x21.Get(i-1, i-1), x21.Vector(i, i-1, 1))
		if i < p {
			*x11.GetPtr(i, i-1), *taup1.GetPtr(i - 1) = Dlarfgp(p-i, x11.Get(i, i-1), x11.Vector(i+2-1, i-1, 1))
			phi.Set(i-1, math.Atan2(x11.Get(i, i-1), x21.Get(i-1, i-1)))
			c = math.Cos(phi.Get(i - 1))
			s = math.Sin(phi.Get(i - 1))
			x11.Set(i, i-1, one)
			Dlarf(Left, p-i, q-i, x11.Vector(i, i-1, 1), taup1.Get(i-1), x11.Off(i, i), work.Off(ilarf-1))
		}
		x21.Set(i-1, i-1, one)
		Dlarf(Left, m-p-i+1, q-i, x21.Vector(i-1, i-1, 1), taup2.Get(i-1), x21.Off(i-1, i), work.Off(ilarf-1))

	}

	//     Reduce the bottom-right portion of X21 to the identity matrix
	for i = p + 1; i <= q; i++ {
		*x21.GetPtr(i-1, i-1), *taup2.GetPtr(i - 1) = Dlarfgp(m-p-i+1, x21.Get(i-1, i-1), x21.Vector(i, i-1, 1))
		x21.Set(i-1, i-1, one)
		Dlarf(Left, m-p-i+1, q-i, x21.Vector(i-1, i-1, 1), taup2.Get(i-1), x21.Off(i-1, i), work.Off(ilarf-1))
	}

	return
}
