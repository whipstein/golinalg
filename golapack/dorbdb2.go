package golapack

import (
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
func Dorbdb2(m, p, q *int, x11 *mat.Matrix, ldx11 *int, x21 *mat.Matrix, ldx21 *int, theta, phi, taup1, taup2, tauq1, work *mat.Vector, lwork, info *int) {
	var lquery bool
	var c, negone, one, s float64
	var childinfo, i, ilarf, iorbdb5, llarf, lorbdb5, lworkmin, lworkopt int

	negone = -1.0
	one = 1.0

	//     Test input arguments
	(*info) = 0
	lquery = (*lwork) == -1

	if (*m) < 0 {
		(*info) = -1
	} else if (*p) < 0 || (*p) > (*m)-(*p) {
		(*info) = -2
	} else if (*q) < 0 || (*q) < (*p) || (*m)-(*q) < (*p) {
		(*info) = -3
	} else if (*ldx11) < maxint(1, *p) {
		(*info) = -5
	} else if (*ldx21) < maxint(1, (*m)-(*p)) {
		(*info) = -7
	}

	//     Compute workspace
	if (*info) == 0 {
		ilarf = 2
		llarf = maxint((*p)-1, (*m)-(*p), (*q)-1)
		iorbdb5 = 2
		lorbdb5 = (*q) - 1
		lworkopt = maxint(ilarf+llarf-1, iorbdb5+lorbdb5-1)
		lworkmin = lworkopt
		work.Set(0, float64(lworkopt))
		if (*lwork) < lworkmin && !lquery {
			(*info) = -14
		}
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DORBDB2"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Reduce rows 1, ..., P of X11 and X21
	for i = 1; i <= (*p); i++ {

		if i > 1 {
			goblas.Drot((*q)-i+1, x11.Vector(i-1, i-1), *ldx11, x21.Vector(i-1-1, i-1), *ldx21, c, s)
		}
		Dlarfgp(toPtr((*q)-i+1), x11.GetPtr(i-1, i-1), x11.Vector(i-1, i+1-1), ldx11, tauq1.GetPtr(i-1))
		c = x11.Get(i-1, i-1)
		x11.Set(i-1, i-1, one)
		Dlarf('R', toPtr((*p)-i), toPtr((*q)-i+1), x11.Vector(i-1, i-1), ldx11, tauq1.GetPtr(i-1), x11.Off(i+1-1, i-1), ldx11, work.Off(ilarf-1))
		Dlarf('R', toPtr((*m)-(*p)-i+1), toPtr((*q)-i+1), x11.Vector(i-1, i-1), ldx11, tauq1.GetPtr(i-1), x21.Off(i-1, i-1), ldx21, work.Off(ilarf-1))
		s = math.Sqrt(math.Pow(goblas.Dnrm2((*p)-i, x11.Vector(i+1-1, i-1), 1), 2) + math.Pow(goblas.Dnrm2((*m)-(*p)-i+1, x21.Vector(i-1, i-1), 1), 2))
		theta.Set(i-1, math.Atan2(s, c))

		Dorbdb5(toPtr((*p)-i), toPtr((*m)-(*p)-i+1), toPtr((*q)-i), x11.Vector(i+1-1, i-1), func() *int { y := 1; return &y }(), x21.Vector(i-1, i-1), func() *int { y := 1; return &y }(), x11.Off(i+1-1, i+1-1), ldx11, x21.Off(i-1, i+1-1), ldx21, work.Off(iorbdb5-1), &lorbdb5, &childinfo)
		goblas.Dscal((*p)-i, negone, x11.Vector(i+1-1, i-1), 1)
		Dlarfgp(toPtr((*m)-(*p)-i+1), x21.GetPtr(i-1, i-1), x21.Vector(i+1-1, i-1), func() *int { y := 1; return &y }(), taup2.GetPtr(i-1))
		if i < (*p) {
			Dlarfgp(toPtr((*p)-i), x11.GetPtr(i+1-1, i-1), x11.Vector(i+2-1, i-1), func() *int { y := 1; return &y }(), taup1.GetPtr(i-1))
			phi.Set(i-1, math.Atan2(x11.Get(i+1-1, i-1), x21.Get(i-1, i-1)))
			c = math.Cos(phi.Get(i - 1))
			s = math.Sin(phi.Get(i - 1))
			x11.Set(i+1-1, i-1, one)
			Dlarf('L', toPtr((*p)-i), toPtr((*q)-i), x11.Vector(i+1-1, i-1), func() *int { y := 1; return &y }(), taup1.GetPtr(i-1), x11.Off(i+1-1, i+1-1), ldx11, work.Off(ilarf-1))
		}
		x21.Set(i-1, i-1, one)
		Dlarf('L', toPtr((*m)-(*p)-i+1), toPtr((*q)-i), x21.Vector(i-1, i-1), func() *int { y := 1; return &y }(), taup2.GetPtr(i-1), x21.Off(i-1, i+1-1), ldx21, work.Off(ilarf-1))

	}

	//     Reduce the bottom-right portion of X21 to the identity matrix
	for i = (*p) + 1; i <= (*q); i++ {
		Dlarfgp(toPtr((*m)-(*p)-i+1), x21.GetPtr(i-1, i-1), x21.Vector(i+1-1, i-1), func() *int { y := 1; return &y }(), taup2.GetPtr(i-1))
		x21.Set(i-1, i-1, one)
		Dlarf('L', toPtr((*m)-(*p)-i+1), toPtr((*q)-i), x21.Vector(i-1, i-1), func() *int { y := 1; return &y }(), taup2.GetPtr(i-1), x21.Off(i-1, i+1-1), ldx21, work.Off(ilarf-1))
	}
}
