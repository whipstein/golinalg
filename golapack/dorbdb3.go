package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dorbdb3 simultaneously bidiagonalizes the blocks of a tall and skinny
// matrix X with orthonomal columns:
//
//                            [ B11 ]
//      [ X11 ]   [ P1 |    ] [  0  ]
//      [-----] = [---------] [-----] Q1**T .
//      [ X21 ]   [    | P2 ] [ B21 ]
//                            [  0  ]
//
// X11 is P-by-Q, and X21 is (M-P)-by-Q. M-P must be no larger than P,
// Q, or M-Q. Routines DORBDB1, DORBDB2, and DORBDB4 handle cases in
// which M-P is not the minimum dimension.
//
// The orthogonal matrices P1, P2, and Q1 are P-by-P, (M-P)-by-(M-P),
// and (M-Q)-by-(M-Q), respectively. They are represented implicitly by
// Householder vectors.
//
// B11 and B12 are (M-P)-by-(M-P) bidiagonal matrices represented
// implicitly by angles THETA, PHI.
func Dorbdb3(m, p, q int, x11, x21 *mat.Matrix, theta, phi, taup1, taup2, tauq1, work *mat.Vector, lwork int) (err error) {
	var lquery bool
	var c, one, s float64
	var i, ilarf, iorbdb5, llarf, lorbdb5, lworkmin, lworkopt int

	one = 1.0

	//     Test input arguments
	lquery = lwork == -1

	if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if 2*p < m || p > m {
		err = fmt.Errorf("2*p < m || p > m: p=%v, m=%v", p, m)
	} else if q < m-p || m-q < m-p {
		err = fmt.Errorf("q < m-p || m-q < m-p: q=%v, m=%v, p=%v", q, m, p)
	} else if x11.Rows < max(1, p) {
		err = fmt.Errorf("x11.Rows < max(1, p): x11.Rows=%v, p=%v", x11.Rows, p)
	} else if x21.Rows < max(1, m-p) {
		err = fmt.Errorf("x21.Rows < max(1, m-p): x21.Rows=%v, m=%v, p=%v", x21.Rows, m, p)
	}

	//     Compute workspace
	if err == nil {
		ilarf = 2
		llarf = max(p, m-p-1, q-1)
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
		gltest.Xerbla2("Dorbdb3", err)
		return
	} else if lquery {
		return
	}

	//     Reduce rows 1, ..., M-P of X11 and X21
	for i = 1; i <= m-p; i++ {

		if i > 1 {
			x21.Off(i-1, i-1).Vector().Rot(q-i+1, x11.Off(i-1-1, i-1).Vector(), x11.Rows, x21.Rows, c, s)
		}

		*x21.GetPtr(i-1, i-1), *tauq1.GetPtr(i - 1) = Dlarfgp(q-i+1, x21.Get(i-1, i-1), x21.Off(i-1, i).Vector(), x21.Rows)
		s = x21.Get(i-1, i-1)
		x21.Set(i-1, i-1, one)
		Dlarf(Right, p-i+1, q-i+1, x21.Off(i-1, i-1).Vector(), x21.Rows, tauq1.Get(i-1), x11.Off(i-1, i-1), work.Off(ilarf-1))
		Dlarf(Right, m-p-i, q-i+1, x21.Off(i-1, i-1).Vector(), x21.Rows, tauq1.Get(i-1), x21.Off(i, i-1), work.Off(ilarf-1))
		c = math.Sqrt(math.Pow(x11.Off(i-1, i-1).Vector().Nrm2(p-i+1, 1), 2) + math.Pow(x21.Off(i, i-1).Vector().Nrm2(m-p-i, 1), 2))
		theta.Set(i-1, math.Atan2(s, c))

		if err = Dorbdb5(p-i+1, m-p-i, q-i, x11.Off(i-1, i-1).Vector(), 1, x21.Off(i, i-1).Vector(), 1, x11.Off(i-1, i), x21.Off(i, i), work.Off(iorbdb5-1), lorbdb5); err != nil {
			panic(err)
		}
		*x11.GetPtr(i-1, i-1), *taup1.GetPtr(i - 1) = Dlarfgp(p-i+1, x11.Get(i-1, i-1), x11.Off(i, i-1).Vector(), 1)
		if i < m-p {
			*x21.GetPtr(i, i-1), *taup2.GetPtr(i - 1) = Dlarfgp(m-p-i, x21.Get(i, i-1), x21.Off(i+2-1, i-1).Vector(), 1)
			phi.Set(i-1, math.Atan2(x21.Get(i, i-1), x11.Get(i-1, i-1)))
			c = math.Cos(phi.Get(i - 1))
			s = math.Sin(phi.Get(i - 1))
			x21.Set(i, i-1, one)
			Dlarf(Left, m-p-i, q-i, x21.Off(i, i-1).Vector(), 1, taup2.Get(i-1), x21.Off(i, i), work.Off(ilarf-1))
		}
		x11.Set(i-1, i-1, one)
		Dlarf(Left, p-i+1, q-i, x11.Off(i-1, i-1).Vector(), 1, taup1.Get(i-1), x11.Off(i-1, i), work.Off(ilarf-1))

	}

	//     Reduce the bottom-right portion of X11 to the identity matrix
	for i = m - p + 1; i <= q; i++ {
		*x11.GetPtr(i-1, i-1), *taup1.GetPtr(i - 1) = Dlarfgp(p-i+1, x11.Get(i-1, i-1), x11.Off(i, i-1).Vector(), 1)
		x11.Set(i-1, i-1, one)
		Dlarf(Left, p-i+1, q-i, x11.Off(i-1, i-1).Vector(), 1, taup1.Get(i-1), x11.Off(i-1, i), work.Off(ilarf-1))
	}

	return
}
