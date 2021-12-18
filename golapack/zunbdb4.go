package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zunbdb4 simultaneously bidiagonalizes the blocks of a tall and skinny
// matrix X with orthonomal columns:
//
//                            [ B11 ]
//      [ X11 ]   [ P1 |    ] [  0  ]
//      [-----] = [---------] [-----] Q1**T .
//      [ X21 ]   [    | P2 ] [ B21 ]
//                            [  0  ]
//
// X11 is P-by-Q, and X21 is (M-P)-by-Q. M-Q must be no larger than P,
// M-P, or Q. Routines ZUNBDB1, ZUNBDB2, and ZUNBDB3 handle cases in
// which M-Q is not the minimum dimension.
//
// The unitary matrices P1, P2, and Q1 are P-by-P, (M-P)-by-(M-P),
// and (M-Q)-by-(M-Q), respectively. They are represented implicitly by
// Householder vectors.
//
// B11 and B12 are (M-Q)-by-(M-Q) bidiagonal matrices represented
// implicitly by angles THETA, PHI.
func Zunbdb4(m, p, q int, x11, x21 *mat.CMatrix, theta, phi *mat.Vector, taup1, taup2, tauq1, phantom, work *mat.CVector, lwork int) (err error) {
	var lquery bool
	var negone, one, zero complex128
	var c, s float64
	var i, ilarf, iorbdb5, j, llarf, lorbdb5, lworkmin, lworkopt int

	negone = (-1.0 + 0.0*1i)
	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	//     Test input arguments
	lquery = lwork == -1

	if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if p < m-q || m-p < m-q {
		err = fmt.Errorf("p < m-q || m-p < m-q: p=%v, m=%v, q=%v", p, m, q)
	} else if q < m-q || q > m {
		err = fmt.Errorf("q < m-q || q > m: q=%v, m=%v", q, m)
	} else if x11.Rows < max(1, p) {
		err = fmt.Errorf("x11.Rows < max(1, p): x11.Rows=%v, p=%v", x11.Rows, p)
	} else if x21.Rows < max(1, m-p) {
		err = fmt.Errorf("x21.Rows < max(1, m-p): x21.Rows=%v, m=%v, p=%v", x21.Rows, m, p)
	}

	//     Compute workspace
	if err == nil {
		ilarf = 2
		llarf = max(q-1, p-1, m-p-1)
		iorbdb5 = 2
		lorbdb5 = q
		lworkopt = ilarf + llarf - 1
		lworkopt = max(lworkopt, iorbdb5+lorbdb5-1)
		lworkmin = lworkopt
		work.SetRe(0, float64(lworkopt))
		if lwork < lworkmin && !lquery {
			err = fmt.Errorf("lwork < lworkmin && !lquery: lwork=%v, lworkmin=%v, lquery=%v", lwork, lworkmin, lquery)
		}
	}
	if err != nil {
		gltest.Xerbla2("Zunbdb4", err)
		return
	} else if lquery {
		return
	}

	//     Reduce columns 1, ..., M-Q of X11 and X21
	for i = 1; i <= m-q; i++ {

		if i == 1 {
			for j = 1; j <= m; j++ {
				phantom.Set(j-1, zero)
			}
			if err = Zunbdb5(p, m-p, q, phantom, 1, phantom.Off(p), 1, x11, x21, work.Off(iorbdb5-1), lorbdb5); err != nil {
				panic(err)
			}
			phantom.Scal(p, negone, 1)
			*phantom.GetPtr(0), *taup1.GetPtr(0) = Zlarfgp(p, phantom.Get(0), phantom.Off(1), 1)
			*phantom.GetPtr(p), *taup2.GetPtr(0) = Zlarfgp(m-p, phantom.Get(p), phantom.Off(p+2-1), 1)
			theta.Set(i-1, math.Atan2(phantom.GetRe(0), phantom.GetRe(p)))
			c = math.Cos(theta.Get(i - 1))
			s = math.Sin(theta.Get(i - 1))
			phantom.Set(0, one)
			phantom.Set(p, one)
			Zlarf(Left, p, q, phantom, 1, taup1.GetConj(0), x11, work.Off(ilarf-1))
			Zlarf(Left, m-p, q, phantom.Off(p), 1, taup2.GetConj(0), x21, work.Off(ilarf-1))
		} else {
			if err = Zunbdb5(p-i+1, m-p-i+1, q-i+1, x11.Off(i-1, i-1-1).CVector(), 1, x21.Off(i-1, i-1-1).CVector(), 1, x11.Off(i-1, i-1), x21.Off(i-1, i-1), work.Off(iorbdb5-1), lorbdb5); err != nil {
				panic(err)
			}
			x11.Off(i-1, i-1-1).CVector().Scal(p-i+1, negone, 1)
			*x11.GetPtr(i-1, i-1-1), *taup1.GetPtr(i - 1) = Zlarfgp(p-i+1, x11.Get(i-1, i-1-1), x11.Off(i, i-1-1).CVector(), 1)
			*x21.GetPtr(i-1, i-1-1), *taup2.GetPtr(i - 1) = Zlarfgp(m-p-i+1, x21.Get(i-1, i-1-1), x21.Off(i, i-1-1).CVector(), 1)
			theta.Set(i-1, math.Atan2(x11.GetRe(i-1, i-1-1), x21.GetRe(i-1, i-1-1)))
			c = math.Cos(theta.Get(i - 1))
			s = math.Sin(theta.Get(i - 1))
			x11.Set(i-1, i-1-1, one)
			x21.Set(i-1, i-1-1, one)
			Zlarf(Left, p-i+1, q-i+1, x11.Off(i-1, i-1-1).CVector(), 1, taup1.GetConj(i-1), x11.Off(i-1, i-1), work.Off(ilarf-1))
			Zlarf(Left, m-p-i+1, q-i+1, x21.Off(i-1, i-1-1).CVector(), 1, taup2.GetConj(i-1), x21.Off(i-1, i-1), work.Off(ilarf-1))
		}

		x21.Off(i-1, i-1).CVector().Drot(q-i+1, x11.Off(i-1, i-1).CVector(), x11.Rows, x21.Rows, s, -c)
		Zlacgv(q-i+1, x21.Off(i-1, i-1).CVector(), x21.Rows)
		*x21.GetPtr(i-1, i-1), *tauq1.GetPtr(i - 1) = Zlarfgp(q-i+1, x21.Get(i-1, i-1), x21.Off(i-1, i).CVector(), x21.Rows)
		c = x21.GetRe(i-1, i-1)
		x21.Set(i-1, i-1, one)
		Zlarf(Right, p-i, q-i+1, x21.Off(i-1, i-1).CVector(), x21.Rows, tauq1.Get(i-1), x11.Off(i, i-1), work.Off(ilarf-1))
		Zlarf(Right, m-p-i, q-i+1, x21.Off(i-1, i-1).CVector(), x21.Rows, tauq1.Get(i-1), x21.Off(i, i-1), work.Off(ilarf-1))
		Zlacgv(q-i+1, x21.Off(i-1, i-1).CVector(), x21.Rows)
		if i < m-q {
			s = math.Sqrt(math.Pow(x11.Off(i, i-1).CVector().Nrm2(p-i, 1), 2) + math.Pow(x21.Off(i, i-1).CVector().Nrm2(m-p-i, 1), 2))
			phi.Set(i-1, math.Atan2(s, c))
		}

	}

	//     Reduce the bottom-right portion of X11 to [ I 0 ]
	for i = m - q + 1; i <= p; i++ {
		Zlacgv(q-i+1, x11.Off(i-1, i-1).CVector(), x11.Rows)
		*x11.GetPtr(i-1, i-1), *tauq1.GetPtr(i - 1) = Zlarfgp(q-i+1, x11.Get(i-1, i-1), x11.Off(i-1, i).CVector(), x11.Rows)
		x11.Set(i-1, i-1, one)
		Zlarf(Right, p-i, q-i+1, x11.Off(i-1, i-1).CVector(), x11.Rows, tauq1.Get(i-1), x11.Off(i, i-1), work.Off(ilarf-1))
		Zlarf(Right, q-p, q-i+1, x11.Off(i-1, i-1).CVector(), x11.Rows, tauq1.Get(i-1), x21.Off(m-q, i-1), work.Off(ilarf-1))
		Zlacgv(q-i+1, x11.Off(i-1, i-1).CVector(), x11.Rows)
	}

	//     Reduce the bottom-right portion of X21 to [ 0 I ]
	for i = p + 1; i <= q; i++ {
		Zlacgv(q-i+1, x21.Off(m-q+i-p-1, i-1).CVector(), x21.Rows)
		*x21.GetPtr(m-q+i-p-1, i-1), *tauq1.GetPtr(i - 1) = Zlarfgp(q-i+1, x21.Get(m-q+i-p-1, i-1), x21.Off(m-q+i-p-1, i).CVector(), x21.Rows)
		x21.Set(m-q+i-p-1, i-1, one)
		Zlarf(Right, q-i, q-i+1, x21.Off(m-q+i-p-1, i-1).CVector(), x21.Rows, tauq1.Get(i-1), x21.Off(m-q+i-p, i-1), work.Off(ilarf-1))
		Zlacgv(q-i+1, x21.Off(m-q+i-p-1, i-1).CVector(), x21.Rows)
	}

	return
}
