package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zunbdb1 simultaneously bidiagonalizes the blocks of a tall and skinny
// matrix X with orthonomal columns:
//
//                            [ B11 ]
//      [ X11 ]   [ P1 |    ] [  0  ]
//      [-----] = [---------] [-----] Q1**T .
//      [ X21 ]   [    | P2 ] [ B21 ]
//                            [  0  ]
//
// X11 is P-by-Q, and X21 is (M-P)-by-Q. Q must be no larger than P,
// M-P, or M-Q. Routines ZUNBDB2, ZUNBDB3, and ZUNBDB4 handle cases in
// which Q is not the minimum dimension.
//
// The unitary matrices P1, P2, and Q1 are P-by-P, (M-P)-by-(M-P),
// and (M-Q)-by-(M-Q), respectively. They are represented implicitly by
// Householder vectors.
//
// B11 and B12 are Q-by-Q bidiagonal matrices represented implicitly by
// angles THETA, PHI.
func Zunbdb1(m, p, q int, x11, x21 *mat.CMatrix, theta, phi *mat.Vector, taup1, taup2, tauq1, work *mat.CVector, lwork int) (err error) {
	var lquery bool
	var one complex128
	var c, s float64
	var i, ilarf, iorbdb5, llarf, lorbdb5, lworkmin, lworkopt int

	one = (1.0 + 0.0*1i)

	//     Test input arguments
	lquery = lwork == -1

	if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if p < q || m-p < q {
		err = fmt.Errorf("p < q || m-p < q: p=%v, q=%v, m=%v", p, q, m)
	} else if q < 0 || m-q < q {
		err = fmt.Errorf("q < 0 || m-q < q: q=%v, m=%v", q, m)
	} else if x11.Rows < max(1, p) {
		err = fmt.Errorf("x11.Rows < max(1, p): x11.Rows=%v, p=%v", x11.Rows, p)
	} else if x21.Rows < max(1, m-p) {
		err = fmt.Errorf("x21.Rows < max(1, m-p): x21.Rows=%v, m=%v, p=%v", x21.Rows, m, p)
	}

	//     Compute workspace
	if err == nil {
		ilarf = 2
		llarf = max(p-1, m-p-1, q-1)
		iorbdb5 = 2
		lorbdb5 = q - 2
		lworkopt = max(ilarf+llarf-1, iorbdb5+lorbdb5-1)
		lworkmin = lworkopt
		work.SetRe(0, float64(lworkopt))
		if lwork < lworkmin && !lquery {
			err = fmt.Errorf("lwork < lworkmin && !lquery: lwork=%v, lworkmin=%v, lquery=%v", lwork, lworkmin, lquery)
		}
	}
	if err != nil {
		gltest.Xerbla2("Zunbdb1", err)
		return
	} else if lquery {
		return
	}

	//     Reduce columns 1, ..., Q of X11 and X21
	for i = 1; i <= q; i++ {

		*x11.GetPtr(i-1, i-1), *taup1.GetPtr(i - 1) = Zlarfgp(p-i+1, x11.Get(i-1, i-1), x11.CVector(i, i-1, 1))
		*x21.GetPtr(i-1, i-1), *taup2.GetPtr(i - 1) = Zlarfgp(m-p-i+1, x21.Get(i-1, i-1), x21.CVector(i, i-1, 1))
		theta.Set(i-1, math.Atan2(x21.GetRe(i-1, i-1), x11.GetRe(i-1, i-1)))
		c = math.Cos(theta.Get(i - 1))
		s = math.Sin(theta.Get(i - 1))
		x11.Set(i-1, i-1, one)
		x21.Set(i-1, i-1, one)
		Zlarf(Left, p-i+1, q-i, x11.CVector(i-1, i-1, 1), taup1.GetConj(i-1), x11.Off(i-1, i), work.Off(ilarf-1))
		Zlarf(Left, m-p-i+1, q-i, x21.CVector(i-1, i-1, 1), taup2.GetConj(i-1), x21.Off(i-1, i), work.Off(ilarf-1))

		if i < q {
			goblas.Zdrot(q-i, x11.CVector(i-1, i), x21.CVector(i-1, i), c, s)
			Zlacgv(q-i, x21.CVector(i-1, i))
			*x21.GetPtr(i-1, i), *tauq1.GetPtr(i - 1) = Zlarfgp(q-i, x21.Get(i-1, i), x21.CVector(i-1, i+2-1))
			s = x21.GetRe(i-1, i)
			x21.Set(i-1, i, one)
			Zlarf(Right, p-i, q-i, x21.CVector(i-1, i), tauq1.Get(i-1), x11.Off(i, i), work.Off(ilarf-1))
			Zlarf(Right, m-p-i, q-i, x21.CVector(i-1, i), tauq1.Get(i-1), x21.Off(i, i), work.Off(ilarf-1))
			Zlacgv(q-i, x21.CVector(i-1, i))
			c = math.Sqrt(math.Pow(goblas.Dznrm2(p-i, x11.CVector(i, i, 1)), 2) + math.Pow(goblas.Dznrm2(m-p-i, x21.CVector(i, i, 1)), 2))
			phi.Set(i-1, math.Atan2(s, c))
			if err = Zunbdb5(p-i, m-p-i, q-i-1, x11.CVector(i, i, 1), x21.CVector(i, i, 1), x11.Off(i, i+2-1), x21.Off(i, i+2-1), work.Off(iorbdb5-1), lorbdb5); err != nil {
				panic(err)
			}
		}

	}

	return
}
