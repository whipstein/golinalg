package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zunbdb simultaneously bidiagonalizes the blocks of an M-by-M
// partitioned unitary matrix X:
//
//                                 [ B11 | B12 0  0 ]
//     [ X11 | X12 ]   [ P1 |    ] [  0  |  0 -I  0 ] [ Q1 |    ]**H
// X = [-----------] = [---------] [----------------] [---------]   .
//     [ X21 | X22 ]   [    | P2 ] [ B21 | B22 0  0 ] [    | Q2 ]
//                                 [  0  |  0  0  I ]
//
// X11 is P-by-Q. Q must be no larger than P, M-P, or M-Q. (If this is
// not the case, then X must be transposed and/or permuted. This can be
// done in constant time using the TRANS and SIGNS options. See ZUNCSD
// for details.)
//
// The unitary matrices P1, P2, Q1, and Q2 are P-by-P, (M-P)-by-
// (M-P), Q-by-Q, and (M-Q)-by-(M-Q), respectively. They are
// represented implicitly by Householder vectors.
//
// B11, B12, B21, and B22 are Q-by-Q bidiagonal matrices represented
// implicitly by angles THETA, PHI.
func Zunbdb(trans, signs byte, m, p, q *int, x11 *mat.CMatrix, ldx11 *int, x12 *mat.CMatrix, ldx12 *int, x21 *mat.CMatrix, ldx21 *int, x22 *mat.CMatrix, ldx22 *int, theta, phi *mat.Vector, taup1, taup2, tauq1, tauq2, work *mat.CVector, lwork, info *int) {
	var colmajor, lquery bool
	var one complex128
	var realone, z1, z2, z3, z4 float64
	var i, lworkmin, lworkopt int

	realone = 1.0
	one = (1.0 + 0.0*1i)

	//     Test input arguments
	(*info) = 0
	colmajor = trans != 'T'
	if signs != 'O' {
		z1 = realone
		z2 = realone
		z3 = realone
		z4 = realone
	} else {
		z1 = realone
		z2 = -realone
		z3 = realone
		z4 = -realone
	}
	lquery = (*lwork) == -1

	if (*m) < 0 {
		(*info) = -3
	} else if (*p) < 0 || (*p) > (*m) {
		(*info) = -4
	} else if (*q) < 0 || (*q) > (*p) || (*q) > (*m)-(*p) || (*q) > (*m)-(*q) {
		(*info) = -5
	} else if colmajor && (*ldx11) < maxint(1, *p) {
		(*info) = -7
	} else if !colmajor && (*ldx11) < maxint(1, *q) {
		(*info) = -7
	} else if colmajor && (*ldx12) < maxint(1, *p) {
		(*info) = -9
	} else if !colmajor && (*ldx12) < maxint(1, (*m)-(*q)) {
		(*info) = -9
	} else if colmajor && (*ldx21) < maxint(1, (*m)-(*p)) {
		(*info) = -11
	} else if !colmajor && (*ldx21) < maxint(1, *q) {
		(*info) = -11
	} else if colmajor && (*ldx22) < maxint(1, (*m)-(*p)) {
		(*info) = -13
	} else if !colmajor && (*ldx22) < maxint(1, (*m)-(*q)) {
		(*info) = -13
	}

	//     Compute workspace
	if (*info) == 0 {
		lworkopt = (*m) - (*q)
		lworkmin = (*m) - (*q)
		work.SetRe(0, float64(lworkopt))
		if (*lwork) < lworkmin && !lquery {
			(*info) = -21
		}
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("xORBDB"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Handle column-major and row-major separately
	if colmajor {
		//        Reduce columns 1, ..., Q of X11, X12, X21, and X22
		for i = 1; i <= (*q); i++ {

			if i == 1 {
				goblas.Zscal((*p)-i+1, complex(z1, 0), x11.CVector(i-1, i-1), 1)
			} else {
				goblas.Zscal((*p)-i+1, complex(z1*math.Cos(phi.Get(i-1-1)), 0), x11.CVector(i-1, i-1), 1)
				goblas.Zaxpy((*p)-i+1, complex(-z1*z3*z4*math.Sin(phi.Get(i-1-1)), 0), x12.CVector(i-1, i-1-1), 1, x11.CVector(i-1, i-1), 1)
			}
			if i == 1 {
				goblas.Zscal((*m)-(*p)-i+1, complex(z2, 0), x21.CVector(i-1, i-1), 1)
			} else {
				goblas.Zscal((*m)-(*p)-i+1, complex(z2*math.Cos(phi.Get(i-1-1)), 0), x21.CVector(i-1, i-1), 1)
				goblas.Zaxpy((*m)-(*p)-i+1, complex(-z2*z3*z4*math.Sin(phi.Get(i-1-1)), 0), x22.CVector(i-1, i-1-1), 1, x21.CVector(i-1, i-1), 1)
			}

			theta.Set(i-1, math.Atan2(goblas.Dznrm2((*m)-(*p)-i+1, x21.CVector(i-1, i-1), 1), goblas.Dznrm2((*p)-i+1, x11.CVector(i-1, i-1), 1)))

			if (*p) > i {
				Zlarfgp(toPtr((*p)-i+1), x11.GetPtr(i-1, i-1), x11.CVector(i+1-1, i-1), func() *int { y := 1; return &y }(), taup1.GetPtr(i-1))
			} else if (*p) == i {
				Zlarfgp(toPtr((*p)-i+1), x11.GetPtr(i-1, i-1), x11.CVector(i-1, i-1), func() *int { y := 1; return &y }(), taup1.GetPtr(i-1))
			}
			x11.Set(i-1, i-1, one)
			if (*m)-(*p) > i {
				Zlarfgp(toPtr((*m)-(*p)-i+1), x21.GetPtr(i-1, i-1), x21.CVector(i+1-1, i-1), func() *int { y := 1; return &y }(), taup2.GetPtr(i-1))
			} else if (*m)-(*p) == i {
				Zlarfgp(toPtr((*m)-(*p)-i+1), x21.GetPtr(i-1, i-1), x21.CVector(i-1, i-1), func() *int { y := 1; return &y }(), taup2.GetPtr(i-1))
			}
			x21.Set(i-1, i-1, one)

			if (*q) > i {
				Zlarf('L', toPtr((*p)-i+1), toPtr((*q)-i), x11.CVector(i-1, i-1), func() *int { y := 1; return &y }(), toPtrc128(taup1.GetConj(i-1)), x11.Off(i-1, i+1-1), ldx11, work)
				Zlarf('L', toPtr((*m)-(*p)-i+1), toPtr((*q)-i), x21.CVector(i-1, i-1), func() *int { y := 1; return &y }(), toPtrc128(taup2.GetConj(i-1)), x21.Off(i-1, i+1-1), ldx21, work)
			}
			if (*m)-(*q)+1 > i {
				Zlarf('L', toPtr((*p)-i+1), toPtr((*m)-(*q)-i+1), x11.CVector(i-1, i-1), func() *int { y := 1; return &y }(), toPtrc128(taup1.GetConj(i-1)), x12.Off(i-1, i-1), ldx12, work)
				Zlarf('L', toPtr((*m)-(*p)-i+1), toPtr((*m)-(*q)-i+1), x21.CVector(i-1, i-1), func() *int { y := 1; return &y }(), toPtrc128(taup2.GetConj(i-1)), x22.Off(i-1, i-1), ldx22, work)
			}

			if i < (*q) {
				goblas.Zscal((*q)-i, complex(-z1*z3*math.Sin(theta.Get(i-1)), 0), x11.CVector(i-1, i+1-1), *ldx11)
				goblas.Zaxpy((*q)-i, complex(z2*z3*math.Cos(theta.Get(i-1)), 0), x21.CVector(i-1, i+1-1), *ldx21, x11.CVector(i-1, i+1-1), *ldx11)
			}
			goblas.Zscal((*m)-(*q)-i+1, complex(-z1*z4*math.Sin(theta.Get(i-1)), 0), x12.CVector(i-1, i-1), *ldx12)
			goblas.Zaxpy((*m)-(*q)-i+1, complex(z2*z4*math.Cos(theta.Get(i-1)), 0), x22.CVector(i-1, i-1), *ldx22, x12.CVector(i-1, i-1), *ldx12)

			if i < (*q) {
				phi.Set(i-1, math.Atan2(goblas.Dznrm2((*q)-i, x11.CVector(i-1, i+1-1), *ldx11), goblas.Dznrm2((*m)-(*q)-i+1, x12.CVector(i-1, i-1), *ldx12)))
			}

			if i < (*q) {
				Zlacgv(toPtr((*q)-i), x11.CVector(i-1, i+1-1), ldx11)
				if i == (*q)-1 {
					Zlarfgp(toPtr((*q)-i), x11.GetPtr(i-1, i+1-1), x11.CVector(i-1, i+1-1), ldx11, tauq1.GetPtr(i-1))
				} else {
					Zlarfgp(toPtr((*q)-i), x11.GetPtr(i-1, i+1-1), x11.CVector(i-1, i+2-1), ldx11, tauq1.GetPtr(i-1))
				}
				x11.Set(i-1, i+1-1, one)
			}
			if (*m)-(*q)+1 > i {
				Zlacgv(toPtr((*m)-(*q)-i+1), x12.CVector(i-1, i-1), ldx12)
				if (*m)-(*q) == i {
					Zlarfgp(toPtr((*m)-(*q)-i+1), x12.GetPtr(i-1, i-1), x12.CVector(i-1, i-1), ldx12, tauq2.GetPtr(i-1))
				} else {
					Zlarfgp(toPtr((*m)-(*q)-i+1), x12.GetPtr(i-1, i-1), x12.CVector(i-1, i+1-1), ldx12, tauq2.GetPtr(i-1))
				}
			}
			x12.Set(i-1, i-1, one)

			if i < (*q) {
				Zlarf('R', toPtr((*p)-i), toPtr((*q)-i), x11.CVector(i-1, i+1-1), ldx11, tauq1.GetPtr(i-1), x11.Off(i+1-1, i+1-1), ldx11, work)
				Zlarf('R', toPtr((*m)-(*p)-i), toPtr((*q)-i), x11.CVector(i-1, i+1-1), ldx11, tauq1.GetPtr(i-1), x21.Off(i+1-1, i+1-1), ldx21, work)
			}
			if (*p) > i {
				Zlarf('R', toPtr((*p)-i), toPtr((*m)-(*q)-i+1), x12.CVector(i-1, i-1), ldx12, tauq2.GetPtr(i-1), x12.Off(i+1-1, i-1), ldx12, work)
			}
			if (*m)-(*p) > i {
				Zlarf('R', toPtr((*m)-(*p)-i), toPtr((*m)-(*q)-i+1), x12.CVector(i-1, i-1), ldx12, tauq2.GetPtr(i-1), x22.Off(i+1-1, i-1), ldx22, work)
			}

			if i < (*q) {
				Zlacgv(toPtr((*q)-i), x11.CVector(i-1, i+1-1), ldx11)
			}
			Zlacgv(toPtr((*m)-(*q)-i+1), x12.CVector(i-1, i-1), ldx12)

		}

		//        Reduce columns Q + 1, ..., P of X12, X22
		for i = (*q) + 1; i <= (*p); i++ {

			goblas.Zscal((*m)-(*q)-i+1, complex(-z1*z4, 0), x12.CVector(i-1, i-1), *ldx12)
			Zlacgv(toPtr((*m)-(*q)-i+1), x12.CVector(i-1, i-1), ldx12)
			if i >= (*m)-(*q) {
				Zlarfgp(toPtr((*m)-(*q)-i+1), x12.GetPtr(i-1, i-1), x12.CVector(i-1, i-1), ldx12, tauq2.GetPtr(i-1))
			} else {
				Zlarfgp(toPtr((*m)-(*q)-i+1), x12.GetPtr(i-1, i-1), x12.CVector(i-1, i+1-1), ldx12, tauq2.GetPtr(i-1))
			}
			x12.Set(i-1, i-1, one)

			if (*p) > i {
				Zlarf('R', toPtr((*p)-i), toPtr((*m)-(*q)-i+1), x12.CVector(i-1, i-1), ldx12, tauq2.GetPtr(i-1), x12.Off(i+1-1, i-1), ldx12, work)
			}
			if (*m)-(*p)-(*q) >= 1 {
				Zlarf('R', toPtr((*m)-(*p)-(*q)), toPtr((*m)-(*q)-i+1), x12.CVector(i-1, i-1), ldx12, tauq2.GetPtr(i-1), x22.Off((*q)+1-1, i-1), ldx22, work)
			}

			Zlacgv(toPtr((*m)-(*q)-i+1), x12.CVector(i-1, i-1), ldx12)

		}

		//        Reduce columns P + 1, ..., M - Q of X12, X22
		for i = 1; i <= (*m)-(*p)-(*q); i++ {

			goblas.Zscal((*m)-(*p)-(*q)-i+1, complex(z2*z4, 0), x22.CVector((*q)+i-1, (*p)+i-1), *ldx22)
			Zlacgv(toPtr((*m)-(*p)-(*q)-i+1), x22.CVector((*q)+i-1, (*p)+i-1), ldx22)
			Zlarfgp(toPtr((*m)-(*p)-(*q)-i+1), x22.GetPtr((*q)+i-1, (*p)+i-1), x22.CVector((*q)+i-1, (*p)+i+1-1), ldx22, tauq2.GetPtr((*p)+i-1))
			x22.Set((*q)+i-1, (*p)+i-1, one)
			Zlarf('R', toPtr((*m)-(*p)-(*q)-i), toPtr((*m)-(*p)-(*q)-i+1), x22.CVector((*q)+i-1, (*p)+i-1), ldx22, tauq2.GetPtr((*p)+i-1), x22.Off((*q)+i+1-1, (*p)+i-1), ldx22, work)

			Zlacgv(toPtr((*m)-(*p)-(*q)-i+1), x22.CVector((*q)+i-1, (*p)+i-1), ldx22)

		}

	} else {
		//        Reduce columns 1, ..., Q of X11, X12, X21, X22
		for i = 1; i <= (*q); i++ {

			if i == 1 {
				goblas.Zscal((*p)-i+1, complex(z1, 0), x11.CVector(i-1, i-1), *ldx11)
			} else {
				goblas.Zscal((*p)-i+1, complex(z1*math.Cos(phi.Get(i-1-1)), 0), x11.CVector(i-1, i-1), *ldx11)
				goblas.Zaxpy((*p)-i+1, complex(-z1*z3*z4*math.Sin(phi.Get(i-1-1)), 0), x12.CVector(i-1-1, i-1), *ldx12, x11.CVector(i-1, i-1), *ldx11)
			}
			if i == 1 {
				goblas.Zscal((*m)-(*p)-i+1, complex(z2, 0), x21.CVector(i-1, i-1), *ldx21)
			} else {
				goblas.Zscal((*m)-(*p)-i+1, complex(z2*math.Cos(phi.Get(i-1-1)), 0), x21.CVector(i-1, i-1), *ldx21)
				goblas.Zaxpy((*m)-(*p)-i+1, complex(-z2*z3*z4*math.Sin(phi.Get(i-1-1)), 0), x22.CVector(i-1-1, i-1), *ldx22, x21.CVector(i-1, i-1), *ldx21)
			}

			theta.Set(i-1, math.Atan2(goblas.Dznrm2((*m)-(*p)-i+1, x21.CVector(i-1, i-1), *ldx21), goblas.Dznrm2((*p)-i+1, x11.CVector(i-1, i-1), *ldx11)))

			Zlacgv(toPtr((*p)-i+1), x11.CVector(i-1, i-1), ldx11)
			Zlacgv(toPtr((*m)-(*p)-i+1), x21.CVector(i-1, i-1), ldx21)

			Zlarfgp(toPtr((*p)-i+1), x11.GetPtr(i-1, i-1), x11.CVector(i-1, i+1-1), ldx11, taup1.GetPtr(i-1))
			x11.Set(i-1, i-1, one)
			if i == (*m)-(*p) {
				Zlarfgp(toPtr((*m)-(*p)-i+1), x21.GetPtr(i-1, i-1), x21.CVector(i-1, i-1), ldx21, taup2.GetPtr(i-1))
			} else {
				Zlarfgp(toPtr((*m)-(*p)-i+1), x21.GetPtr(i-1, i-1), x21.CVector(i-1, i+1-1), ldx21, taup2.GetPtr(i-1))
			}
			x21.Set(i-1, i-1, one)

			Zlarf('R', toPtr((*q)-i), toPtr((*p)-i+1), x11.CVector(i-1, i-1), ldx11, taup1.GetPtr(i-1), x11.Off(i+1-1, i-1), ldx11, work)
			Zlarf('R', toPtr((*m)-(*q)-i+1), toPtr((*p)-i+1), x11.CVector(i-1, i-1), ldx11, taup1.GetPtr(i-1), x12.Off(i-1, i-1), ldx12, work)
			Zlarf('R', toPtr((*q)-i), toPtr((*m)-(*p)-i+1), x21.CVector(i-1, i-1), ldx21, taup2.GetPtr(i-1), x21.Off(i+1-1, i-1), ldx21, work)
			Zlarf('R', toPtr((*m)-(*q)-i+1), toPtr((*m)-(*p)-i+1), x21.CVector(i-1, i-1), ldx21, taup2.GetPtr(i-1), x22.Off(i-1, i-1), ldx22, work)

			Zlacgv(toPtr((*p)-i+1), x11.CVector(i-1, i-1), ldx11)
			Zlacgv(toPtr((*m)-(*p)-i+1), x21.CVector(i-1, i-1), ldx21)

			if i < (*q) {
				goblas.Zscal((*q)-i, complex(-z1*z3*math.Sin(theta.Get(i-1)), 0), x11.CVector(i+1-1, i-1), 1)
				goblas.Zaxpy((*q)-i, complex(z2*z3*math.Cos(theta.Get(i-1)), 0), x21.CVector(i+1-1, i-1), 1, x11.CVector(i+1-1, i-1), 1)
			}
			goblas.Zscal((*m)-(*q)-i+1, complex(-z1*z4*math.Sin(theta.Get(i-1)), 0), x12.CVector(i-1, i-1), 1)
			goblas.Zaxpy((*m)-(*q)-i+1, complex(z2*z4*math.Cos(theta.Get(i-1)), 0), x22.CVector(i-1, i-1), 1, x12.CVector(i-1, i-1), 1)

			if i < (*q) {
				phi.Set(i-1, math.Atan2(goblas.Dznrm2((*q)-i, x11.CVector(i+1-1, i-1), 1), goblas.Dznrm2((*m)-(*q)-i+1, x12.CVector(i-1, i-1), 1)))
			}

			if i < (*q) {
				Zlarfgp(toPtr((*q)-i), x11.GetPtr(i+1-1, i-1), x11.CVector(i+2-1, i-1), func() *int { y := 1; return &y }(), tauq1.GetPtr(i-1))
				x11.Set(i+1-1, i-1, one)
			}
			Zlarfgp(toPtr((*m)-(*q)-i+1), x12.GetPtr(i-1, i-1), x12.CVector(i+1-1, i-1), func() *int { y := 1; return &y }(), tauq2.GetPtr(i-1))
			x12.Set(i-1, i-1, one)

			if i < (*q) {
				Zlarf('L', toPtr((*q)-i), toPtr((*p)-i), x11.CVector(i+1-1, i-1), func() *int { y := 1; return &y }(), toPtrc128(tauq1.GetConj(i-1)), x11.Off(i+1-1, i+1-1), ldx11, work)
				Zlarf('L', toPtr((*q)-i), toPtr((*m)-(*p)-i), x11.CVector(i+1-1, i-1), func() *int { y := 1; return &y }(), toPtrc128(tauq1.GetConj(i-1)), x21.Off(i+1-1, i+1-1), ldx21, work)
			}
			Zlarf('L', toPtr((*m)-(*q)-i+1), toPtr((*p)-i), x12.CVector(i-1, i-1), func() *int { y := 1; return &y }(), toPtrc128(tauq2.GetConj(i-1)), x12.Off(i-1, i+1-1), ldx12, work)
			if (*m)-(*p) > i {
				Zlarf('L', toPtr((*m)-(*q)-i+1), toPtr((*m)-(*p)-i), x12.CVector(i-1, i-1), func() *int { y := 1; return &y }(), toPtrc128(tauq2.GetConj(i-1)), x22.Off(i-1, i+1-1), ldx22, work)
			}

		}

		//        Reduce columns Q + 1, ..., P of X12, X22
		for i = (*q) + 1; i <= (*p); i++ {

			goblas.Zscal((*m)-(*q)-i+1, complex(-z1*z4, 0), x12.CVector(i-1, i-1), 1)
			Zlarfgp(toPtr((*m)-(*q)-i+1), x12.GetPtr(i-1, i-1), x12.CVector(i+1-1, i-1), func() *int { y := 1; return &y }(), tauq2.GetPtr(i-1))
			x12.Set(i-1, i-1, one)

			if (*p) > i {
				Zlarf('L', toPtr((*m)-(*q)-i+1), toPtr((*p)-i), x12.CVector(i-1, i-1), func() *int { y := 1; return &y }(), toPtrc128(tauq2.GetConj(i-1)), x12.Off(i-1, i+1-1), ldx12, work)
			}
			if (*m)-(*p)-(*q) >= 1 {
				Zlarf('L', toPtr((*m)-(*q)-i+1), toPtr((*m)-(*p)-(*q)), x12.CVector(i-1, i-1), func() *int { y := 1; return &y }(), toPtrc128(tauq2.GetConj(i-1)), x22.Off(i-1, (*q)+1-1), ldx22, work)
			}

		}

		//        Reduce columns P + 1, ..., M - Q of X12, X22
		for i = 1; i <= (*m)-(*p)-(*q); i++ {

			goblas.Zscal((*m)-(*p)-(*q)-i+1, complex(z2*z4, 0), x22.CVector((*p)+i-1, (*q)+i-1), 1)
			Zlarfgp(toPtr((*m)-(*p)-(*q)-i+1), x22.GetPtr((*p)+i-1, (*q)+i-1), x22.CVector((*p)+i+1-1, (*q)+i-1), func() *int { y := 1; return &y }(), tauq2.GetPtr((*p)+i-1))
			x22.Set((*p)+i-1, (*q)+i-1, one)

			if (*m)-(*p)-(*q) != i {
				Zlarf('L', toPtr((*m)-(*p)-(*q)-i+1), toPtr((*m)-(*p)-(*q)-i), x22.CVector((*p)+i-1, (*q)+i-1), func() *int { y := 1; return &y }(), toPtrc128(tauq2.GetConj((*p)+i-1)), x22.Off((*p)+i-1, (*q)+i+1-1), ldx22, work)
			}

		}

	}
}
