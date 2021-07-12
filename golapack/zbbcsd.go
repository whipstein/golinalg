package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zbbcsd computes the CS decomposition of a unitary matrix in
// bidiagonal-block form,
//
//
//     [ B11 | B12 0  0 ]
//     [  0  |  0 -I  0 ]
// X = [----------------]
//     [ B21 | B22 0  0 ]
//     [  0  |  0  0  I ]
//
//                               [  C | -S  0  0 ]
//                   [ U1 |    ] [  0 |  0 -I  0 ] [ V1 |    ]**H
//                 = [---------] [---------------] [---------]   .
//                   [    | U2 ] [  S |  C  0  0 ] [    | V2 ]
//                               [  0 |  0  0  I ]
//
// X is M-by-M, its top-left block is P-by-Q, and Q must be no larger
// than P, M-P, or M-Q. (If Q is not the smallest index, then X must be
// transposed and/or permuted. This can be done in constant time using
// the TRANS and SIGNS options. See ZUNCSD for details.)
//
// The bidiagonal matrices B11, B12, B21, and B22 are represented
// implicitly by angles THETA(1:Q) and PHI(1:Q-1).
//
// The unitary matrices U1, U2, V1T, and V2T are input/output.
// The input matrices are pre- or post-multiplied by the appropriate
// singular vector matrices.
func Zbbcsd(jobu1, jobu2, jobv1t, jobv2t, trans byte, m, p, q *int, theta, phi *mat.Vector, u1 *mat.CMatrix, ldu1 *int, u2 *mat.CMatrix, ldu2 *int, v1t *mat.CMatrix, ldv1t *int, v2t *mat.CMatrix, ldv2t *int, b11d, b11e, b12d, b12e, b21d, b21e, b22d, b22e, rwork *mat.Vector, lrwork, info *int) {
	var colmajor, lquery, restart11, restart12, restart21, restart22, wantu1, wantu2, wantv1t, wantv2t bool
	var negonecomplex complex128
	var b11bulge, b12bulge, b21bulge, b22bulge, dummy, eps, hundred, meighth, mu, nu, one, piover2, r, sigma11, sigma21, temp, ten, thetamax, thetamin, thresh, tol, tolmul, unfl, x1, x2, y1, y2, zero float64
	var i, imax, imin, iter, iu1cs, iu1sn, iu2cs, iu2sn, iv1tcs, iv1tsn, iv2tcs, iv2tsn, j, lrworkmin, lrworkopt, maxit, maxitr, mini int

	maxitr = 6
	hundred = 100.0
	meighth = -0.125
	one = 1.0
	piover2 = 1.57079632679489662
	ten = 10.0
	zero = 0.0
	negonecomplex = (-1.0 + 0.0*1i)

	//     Test input arguments
	(*info) = 0
	lquery = (*lrwork) == -1
	wantu1 = jobu1 == 'Y'
	wantu2 = jobu2 == 'Y'
	wantv1t = jobv1t == 'Y'
	wantv2t = jobv2t == 'Y'
	colmajor = trans != 'T'

	if (*m) < 0 {
		(*info) = -6
	} else if (*p) < 0 || (*p) > (*m) {
		(*info) = -7
	} else if (*q) < 0 || (*q) > (*m) {
		(*info) = -8
	} else if (*q) > (*p) || (*q) > (*m)-(*p) || (*q) > (*m)-(*q) {
		(*info) = -8
	} else if wantu1 && (*ldu1) < (*p) {
		(*info) = -12
	} else if wantu2 && (*ldu2) < (*m)-(*p) {
		(*info) = -14
	} else if wantv1t && (*ldv1t) < (*q) {
		(*info) = -16
	} else if wantv2t && (*ldv2t) < (*m)-(*q) {
		(*info) = -18
	}

	//     Quick return if Q = 0
	if (*info) == 0 && (*q) == 0 {
		lrworkmin = 1
		rwork.Set(0, float64(lrworkmin))
		return
	}

	//     Compute workspace
	if (*info) == 0 {
		iu1cs = 1
		iu1sn = iu1cs + (*q)
		iu2cs = iu1sn + (*q)
		iu2sn = iu2cs + (*q)
		iv1tcs = iu2sn + (*q)
		iv1tsn = iv1tcs + (*q)
		iv2tcs = iv1tsn + (*q)
		iv2tsn = iv2tcs + (*q)
		lrworkopt = iv2tsn + (*q) - 1
		lrworkmin = lrworkopt
		rwork.Set(0, float64(lrworkopt))
		if (*lrwork) < lrworkmin && !lquery {
			(*info) = -28
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZBBCSD"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Get machine constants
	eps = Dlamch(Epsilon)
	unfl = Dlamch(SafeMinimum)
	tolmul = math.Max(ten, math.Min(hundred, math.Pow(eps, meighth)))
	tol = tolmul * eps
	thresh = math.Max(tol, float64(maxitr*(*q)*(*q))*unfl)

	//     Test for negligible sines or cosines
	for i = 1; i <= (*q); i++ {
		if theta.Get(i-1) < thresh {
			theta.Set(i-1, zero)
		} else if theta.Get(i-1) > piover2-thresh {
			theta.Set(i-1, piover2)
		}
	}
	for i = 1; i <= (*q)-1; i++ {
		if phi.Get(i-1) < thresh {
			phi.Set(i-1, zero)
		} else if phi.Get(i-1) > piover2-thresh {
			phi.Set(i-1, piover2)
		}
	}

	//     Initial deflation
	imax = (*q)
	for imax > 1 {
		if phi.Get(imax-1-1) != zero {
			break
		}
		imax = imax - 1
	}
	imin = imax - 1
	if imin > 1 {
		for phi.Get(imin-1-1) != zero {
			imin = imin - 1
			if imin <= 1 {
				break
			}
		}
	}

	//     Initialize iteration counter
	maxit = maxitr * (*q) * (*q)
	iter = 0

	//     Begin main iteration loop
	for imax > 1 {
		//        Compute the matrix entries
		b11d.Set(imin-1, math.Cos(theta.Get(imin-1)))
		b21d.Set(imin-1, -math.Sin(theta.Get(imin-1)))
		for i = imin; i <= imax-1; i++ {
			b11e.Set(i-1, -math.Sin(theta.Get(i-1))*math.Sin(phi.Get(i-1)))
			b11d.Set(i, math.Cos(theta.Get(i))*math.Cos(phi.Get(i-1)))
			b12d.Set(i-1, math.Sin(theta.Get(i-1))*math.Cos(phi.Get(i-1)))
			b12e.Set(i-1, math.Cos(theta.Get(i))*math.Sin(phi.Get(i-1)))
			b21e.Set(i-1, -math.Cos(theta.Get(i-1))*math.Sin(phi.Get(i-1)))
			b21d.Set(i, -math.Sin(theta.Get(i))*math.Cos(phi.Get(i-1)))
			b22d.Set(i-1, math.Cos(theta.Get(i-1))*math.Cos(phi.Get(i-1)))
			b22e.Set(i-1, -math.Sin(theta.Get(i))*math.Sin(phi.Get(i-1)))
		}
		b12d.Set(imax-1, math.Sin(theta.Get(imax-1)))
		b22d.Set(imax-1, math.Cos(theta.Get(imax-1)))

		//        Abort if not converging; otherwise, increment ITER
		if iter > maxit {
			(*info) = 0
			for i = 1; i <= (*q); i++ {
				if phi.Get(i-1) != zero {
					(*info) = (*info) + 1
				}
			}
			return
		}

		iter = iter + imax - imin

		//        Compute shifts
		thetamax = theta.Get(imin - 1)
		thetamin = theta.Get(imin - 1)
		for i = imin + 1; i <= imax; i++ {
			if theta.Get(i-1) > thetamax {
				thetamax = theta.Get(i - 1)
			}
			if theta.Get(i-1) < thetamin {
				thetamin = theta.Get(i - 1)
			}
		}

		if thetamax > piover2-thresh {
			//           Zero on diagonals of B11 and B22; induce deflation with a
			//           zero shift
			mu = zero
			nu = one

		} else if thetamin < thresh {
			//           Zero on diagonals of B12 and B22; induce deflation with a
			//           zero shift
			mu = one
			nu = zero

		} else {
			//           Compute shifts for B11 and B21 and use the lesser
			Dlas2(b11d.GetPtr(imax-1-1), b11e.GetPtr(imax-1-1), b11d.GetPtr(imax-1), &sigma11, &dummy)
			Dlas2(b21d.GetPtr(imax-1-1), b21e.GetPtr(imax-1-1), b21d.GetPtr(imax-1), &sigma21, &dummy)

			if sigma11 <= sigma21 {
				mu = sigma11
				nu = math.Sqrt(one - math.Pow(mu, 2))
				if mu < thresh {
					mu = zero
					nu = one
				}
			} else {
				nu = sigma21
				mu = math.Sqrt(1.0 - math.Pow(nu, 2))
				if nu < thresh {
					mu = one
					nu = zero
				}
			}
		}

		//        Rotate to produce bulges in B11 and B21
		if mu <= nu {
			Dlartgs(b11d.GetPtr(imin-1), b11e.GetPtr(imin-1), &mu, rwork.GetPtr(iv1tcs+imin-1-1), rwork.GetPtr(iv1tsn+imin-1-1))
		} else {
			Dlartgs(b21d.GetPtr(imin-1), b21e.GetPtr(imin-1), &nu, rwork.GetPtr(iv1tcs+imin-1-1), rwork.GetPtr(iv1tsn+imin-1-1))
		}

		temp = rwork.Get(iv1tcs+imin-1-1)*b11d.Get(imin-1) + rwork.Get(iv1tsn+imin-1-1)*b11e.Get(imin-1)
		b11e.Set(imin-1, rwork.Get(iv1tcs+imin-1-1)*b11e.Get(imin-1)-rwork.Get(iv1tsn+imin-1-1)*b11d.Get(imin-1))
		b11d.Set(imin-1, temp)
		b11bulge = rwork.Get(iv1tsn+imin-1-1) * b11d.Get(imin)
		b11d.Set(imin, rwork.Get(iv1tcs+imin-1-1)*b11d.Get(imin))
		temp = rwork.Get(iv1tcs+imin-1-1)*b21d.Get(imin-1) + rwork.Get(iv1tsn+imin-1-1)*b21e.Get(imin-1)
		b21e.Set(imin-1, rwork.Get(iv1tcs+imin-1-1)*b21e.Get(imin-1)-rwork.Get(iv1tsn+imin-1-1)*b21d.Get(imin-1))
		b21d.Set(imin-1, temp)
		b21bulge = rwork.Get(iv1tsn+imin-1-1) * b21d.Get(imin)
		b21d.Set(imin, rwork.Get(iv1tcs+imin-1-1)*b21d.Get(imin))

		//        Compute THETA(IMIN)
		theta.Set(imin-1, math.Atan2(math.Sqrt(math.Pow(b21d.Get(imin-1), 2)+math.Pow(b21bulge, 2)), math.Sqrt(math.Pow(b11d.Get(imin-1), 2)+math.Pow(b11bulge, 2))))

		//        Chase the bulges in B11(IMIN+1,IMIN) and B21(IMIN+1,IMIN)
		if math.Pow(b11d.Get(imin-1), 2)+math.Pow(b11bulge, 2) > math.Pow(thresh, 2) {
			Dlartgp(&b11bulge, b11d.GetPtr(imin-1), rwork.GetPtr(iu1sn+imin-1-1), rwork.GetPtr(iu1cs+imin-1-1), &r)
		} else if mu <= nu {
			Dlartgs(b11e.GetPtr(imin-1), b11d.GetPtr(imin), &mu, rwork.GetPtr(iu1cs+imin-1-1), rwork.GetPtr(iu1sn+imin-1-1))
		} else {
			Dlartgs(b12d.GetPtr(imin-1), b12e.GetPtr(imin-1), &nu, rwork.GetPtr(iu1cs+imin-1-1), rwork.GetPtr(iu1sn+imin-1-1))
		}
		if math.Pow(b21d.Get(imin-1), 2)+math.Pow(b21bulge, 2) > math.Pow(thresh, 2) {
			Dlartgp(&b21bulge, b21d.GetPtr(imin-1), rwork.GetPtr(iu2sn+imin-1-1), rwork.GetPtr(iu2cs+imin-1-1), &r)
		} else if nu < mu {
			Dlartgs(b21e.GetPtr(imin-1), b21d.GetPtr(imin), &nu, rwork.GetPtr(iu2cs+imin-1-1), rwork.GetPtr(iu2sn+imin-1-1))
		} else {
			Dlartgs(b22d.GetPtr(imin-1), b22e.GetPtr(imin-1), &mu, rwork.GetPtr(iu2cs+imin-1-1), rwork.GetPtr(iu2sn+imin-1-1))
		}
		rwork.Set(iu2cs+imin-1-1, -rwork.Get(iu2cs+imin-1-1))
		rwork.Set(iu2sn+imin-1-1, -rwork.Get(iu2sn+imin-1-1))

		temp = rwork.Get(iu1cs+imin-1-1)*b11e.Get(imin-1) + rwork.Get(iu1sn+imin-1-1)*b11d.Get(imin)
		b11d.Set(imin, rwork.Get(iu1cs+imin-1-1)*b11d.Get(imin)-rwork.Get(iu1sn+imin-1-1)*b11e.Get(imin-1))
		b11e.Set(imin-1, temp)
		if imax > imin+1 {
			b11bulge = rwork.Get(iu1sn+imin-1-1) * b11e.Get(imin)
			b11e.Set(imin, rwork.Get(iu1cs+imin-1-1)*b11e.Get(imin))
		}
		temp = rwork.Get(iu1cs+imin-1-1)*b12d.Get(imin-1) + rwork.Get(iu1sn+imin-1-1)*b12e.Get(imin-1)
		b12e.Set(imin-1, rwork.Get(iu1cs+imin-1-1)*b12e.Get(imin-1)-rwork.Get(iu1sn+imin-1-1)*b12d.Get(imin-1))
		b12d.Set(imin-1, temp)
		b12bulge = rwork.Get(iu1sn+imin-1-1) * b12d.Get(imin)
		b12d.Set(imin, rwork.Get(iu1cs+imin-1-1)*b12d.Get(imin))
		temp = rwork.Get(iu2cs+imin-1-1)*b21e.Get(imin-1) + rwork.Get(iu2sn+imin-1-1)*b21d.Get(imin)
		b21d.Set(imin, rwork.Get(iu2cs+imin-1-1)*b21d.Get(imin)-rwork.Get(iu2sn+imin-1-1)*b21e.Get(imin-1))
		b21e.Set(imin-1, temp)
		if imax > imin+1 {
			b21bulge = rwork.Get(iu2sn+imin-1-1) * b21e.Get(imin)
			b21e.Set(imin, rwork.Get(iu2cs+imin-1-1)*b21e.Get(imin))
		}
		temp = rwork.Get(iu2cs+imin-1-1)*b22d.Get(imin-1) + rwork.Get(iu2sn+imin-1-1)*b22e.Get(imin-1)
		b22e.Set(imin-1, rwork.Get(iu2cs+imin-1-1)*b22e.Get(imin-1)-rwork.Get(iu2sn+imin-1-1)*b22d.Get(imin-1))
		b22d.Set(imin-1, temp)
		b22bulge = rwork.Get(iu2sn+imin-1-1) * b22d.Get(imin)
		b22d.Set(imin, rwork.Get(iu2cs+imin-1-1)*b22d.Get(imin))

		//        Inner loop: chase bulges from B11(IMIN,IMIN+2),
		//        B12(IMIN,IMIN+1), B21(IMIN,IMIN+2), and B22(IMIN,IMIN+1) to
		//        bottom-right
		for i = imin + 1; i <= imax-1; i++ {
			//           Compute PHI(I-1)
			x1 = math.Sin(theta.Get(i-1-1))*b11e.Get(i-1-1) + math.Cos(theta.Get(i-1-1))*b21e.Get(i-1-1)
			x2 = math.Sin(theta.Get(i-1-1))*b11bulge + math.Cos(theta.Get(i-1-1))*b21bulge
			y1 = math.Sin(theta.Get(i-1-1))*b12d.Get(i-1-1) + math.Cos(theta.Get(i-1-1))*b22d.Get(i-1-1)
			y2 = math.Sin(theta.Get(i-1-1))*b12bulge + math.Cos(theta.Get(i-1-1))*b22bulge

			phi.Set(i-1-1, math.Atan2(math.Sqrt(math.Pow(x1, 2)+math.Pow(x2, 2)), math.Sqrt(math.Pow(y1, 2)+math.Pow(y2, 2))))

			//           Determine if there are bulges to chase or if a new direct
			//           summand has been reached
			restart11 = math.Pow(b11e.Get(i-1-1), 2)+math.Pow(b11bulge, 2) <= math.Pow(thresh, 2)
			restart21 = math.Pow(b21e.Get(i-1-1), 2)+math.Pow(b21bulge, 2) <= math.Pow(thresh, 2)
			restart12 = math.Pow(b12d.Get(i-1-1), 2)+math.Pow(b12bulge, 2) <= math.Pow(thresh, 2)
			restart22 = math.Pow(b22d.Get(i-1-1), 2)+math.Pow(b22bulge, 2) <= math.Pow(thresh, 2)

			//           If possible, chase bulges from B11(I-1,I+1), B12(I-1,I),
			//           B21(I-1,I+1), and B22(I-1,I). If necessary, restart bulge-
			//           chasing by applying the original shift again.
			if !restart11 && !restart21 {
				Dlartgp(&x2, &x1, rwork.GetPtr(iv1tsn+i-1-1), rwork.GetPtr(iv1tcs+i-1-1), &r)
			} else if !restart11 && restart21 {
				Dlartgp(&b11bulge, b11e.GetPtr(i-1-1), rwork.GetPtr(iv1tsn+i-1-1), rwork.GetPtr(iv1tcs+i-1-1), &r)
			} else if restart11 && !restart21 {
				Dlartgp(&b21bulge, b21e.GetPtr(i-1-1), rwork.GetPtr(iv1tsn+i-1-1), rwork.GetPtr(iv1tcs+i-1-1), &r)
			} else if mu <= nu {
				Dlartgs(b11d.GetPtr(i-1), b11e.GetPtr(i-1), &mu, rwork.GetPtr(iv1tcs+i-1-1), rwork.GetPtr(iv1tsn+i-1-1))
			} else {
				Dlartgs(b21d.GetPtr(i-1), b21e.GetPtr(i-1), &nu, rwork.GetPtr(iv1tcs+i-1-1), rwork.GetPtr(iv1tsn+i-1-1))
			}
			rwork.Set(iv1tcs+i-1-1, -rwork.Get(iv1tcs+i-1-1))
			rwork.Set(iv1tsn+i-1-1, -rwork.Get(iv1tsn+i-1-1))
			if !restart12 && !restart22 {
				Dlartgp(&y2, &y1, rwork.GetPtr(iv2tsn+i-1-1-1), rwork.GetPtr(iv2tcs+i-1-1-1), &r)
			} else if !restart12 && restart22 {
				Dlartgp(&b12bulge, b12d.GetPtr(i-1-1), rwork.GetPtr(iv2tsn+i-1-1-1), rwork.GetPtr(iv2tcs+i-1-1-1), &r)
			} else if restart12 && !restart22 {
				Dlartgp(&b22bulge, b22d.GetPtr(i-1-1), rwork.GetPtr(iv2tsn+i-1-1-1), rwork.GetPtr(iv2tcs+i-1-1-1), &r)
			} else if nu < mu {
				Dlartgs(b12e.GetPtr(i-1-1), b12d.GetPtr(i-1), &nu, rwork.GetPtr(iv2tcs+i-1-1-1), rwork.GetPtr(iv2tsn+i-1-1-1))
			} else {
				Dlartgs(b22e.GetPtr(i-1-1), b22d.GetPtr(i-1), &mu, rwork.GetPtr(iv2tcs+i-1-1-1), rwork.GetPtr(iv2tsn+i-1-1-1))
			}

			temp = rwork.Get(iv1tcs+i-1-1)*b11d.Get(i-1) + rwork.Get(iv1tsn+i-1-1)*b11e.Get(i-1)
			b11e.Set(i-1, rwork.Get(iv1tcs+i-1-1)*b11e.Get(i-1)-rwork.Get(iv1tsn+i-1-1)*b11d.Get(i-1))
			b11d.Set(i-1, temp)
			b11bulge = rwork.Get(iv1tsn+i-1-1) * b11d.Get(i)
			b11d.Set(i, rwork.Get(iv1tcs+i-1-1)*b11d.Get(i))
			temp = rwork.Get(iv1tcs+i-1-1)*b21d.Get(i-1) + rwork.Get(iv1tsn+i-1-1)*b21e.Get(i-1)
			b21e.Set(i-1, rwork.Get(iv1tcs+i-1-1)*b21e.Get(i-1)-rwork.Get(iv1tsn+i-1-1)*b21d.Get(i-1))
			b21d.Set(i-1, temp)
			b21bulge = rwork.Get(iv1tsn+i-1-1) * b21d.Get(i)
			b21d.Set(i, rwork.Get(iv1tcs+i-1-1)*b21d.Get(i))
			temp = rwork.Get(iv2tcs+i-1-1-1)*b12e.Get(i-1-1) + rwork.Get(iv2tsn+i-1-1-1)*b12d.Get(i-1)
			b12d.Set(i-1, rwork.Get(iv2tcs+i-1-1-1)*b12d.Get(i-1)-rwork.Get(iv2tsn+i-1-1-1)*b12e.Get(i-1-1))
			b12e.Set(i-1-1, temp)
			b12bulge = rwork.Get(iv2tsn+i-1-1-1) * b12e.Get(i-1)
			b12e.Set(i-1, rwork.Get(iv2tcs+i-1-1-1)*b12e.Get(i-1))
			temp = rwork.Get(iv2tcs+i-1-1-1)*b22e.Get(i-1-1) + rwork.Get(iv2tsn+i-1-1-1)*b22d.Get(i-1)
			b22d.Set(i-1, rwork.Get(iv2tcs+i-1-1-1)*b22d.Get(i-1)-rwork.Get(iv2tsn+i-1-1-1)*b22e.Get(i-1-1))
			b22e.Set(i-1-1, temp)
			b22bulge = rwork.Get(iv2tsn+i-1-1-1) * b22e.Get(i-1)
			b22e.Set(i-1, rwork.Get(iv2tcs+i-1-1-1)*b22e.Get(i-1))

			//           Compute THETA(I)
			x1 = math.Cos(phi.Get(i-1-1))*b11d.Get(i-1) + math.Sin(phi.Get(i-1-1))*b12e.Get(i-1-1)
			x2 = math.Cos(phi.Get(i-1-1))*b11bulge + math.Sin(phi.Get(i-1-1))*b12bulge
			y1 = math.Cos(phi.Get(i-1-1))*b21d.Get(i-1) + math.Sin(phi.Get(i-1-1))*b22e.Get(i-1-1)
			y2 = math.Cos(phi.Get(i-1-1))*b21bulge + math.Sin(phi.Get(i-1-1))*b22bulge

			theta.Set(i-1, math.Atan2(math.Sqrt(math.Pow(y1, 2)+math.Pow(y2, 2)), math.Sqrt(math.Pow(x1, 2)+math.Pow(x2, 2))))

			//           Determine if there are bulges to chase or if a new direct
			//           summand has been reached
			restart11 = math.Pow(b11d.Get(i-1), 2)+math.Pow(b11bulge, 2) <= math.Pow(thresh, 2)
			restart12 = math.Pow(b12e.Get(i-1-1), 2)+math.Pow(b12bulge, 2) <= math.Pow(thresh, 2)
			restart21 = math.Pow(b21d.Get(i-1), 2)+math.Pow(b21bulge, 2) <= math.Pow(thresh, 2)
			restart22 = math.Pow(b22e.Get(i-1-1), 2)+math.Pow(b22bulge, 2) <= math.Pow(thresh, 2)

			//           If possible, chase bulges from B11(I+1,I), B12(I+1,I-1),
			//           B21(I+1,I), and B22(I+1,I-1). If necessary, restart bulge-
			//           chasing by applying the original shift again.
			if !restart11 && !restart12 {
				Dlartgp(&x2, &x1, rwork.GetPtr(iu1sn+i-1-1), rwork.GetPtr(iu1cs+i-1-1), &r)
			} else if !restart11 && restart12 {
				Dlartgp(&b11bulge, b11d.GetPtr(i-1), rwork.GetPtr(iu1sn+i-1-1), rwork.GetPtr(iu1cs+i-1-1), &r)
			} else if restart11 && !restart12 {
				Dlartgp(&b12bulge, b12e.GetPtr(i-1-1), rwork.GetPtr(iu1sn+i-1-1), rwork.GetPtr(iu1cs+i-1-1), &r)
			} else if mu <= nu {
				Dlartgs(b11e.GetPtr(i-1), b11d.GetPtr(i), &mu, rwork.GetPtr(iu1cs+i-1-1), rwork.GetPtr(iu1sn+i-1-1))
			} else {
				Dlartgs(b12d.GetPtr(i-1), b12e.GetPtr(i-1), &nu, rwork.GetPtr(iu1cs+i-1-1), rwork.GetPtr(iu1sn+i-1-1))
			}
			if !restart21 && !restart22 {
				Dlartgp(&y2, &y1, rwork.GetPtr(iu2sn+i-1-1), rwork.GetPtr(iu2cs+i-1-1), &r)
			} else if !restart21 && restart22 {
				Dlartgp(&b21bulge, b21d.GetPtr(i-1), rwork.GetPtr(iu2sn+i-1-1), rwork.GetPtr(iu2cs+i-1-1), &r)
			} else if restart21 && !restart22 {
				Dlartgp(&b22bulge, b22e.GetPtr(i-1-1), rwork.GetPtr(iu2sn+i-1-1), rwork.GetPtr(iu2cs+i-1-1), &r)
			} else if nu < mu {
				Dlartgs(b21e.GetPtr(i-1), b21e.GetPtr(i), &nu, rwork.GetPtr(iu2cs+i-1-1), rwork.GetPtr(iu2sn+i-1-1))
			} else {
				Dlartgs(b22d.GetPtr(i-1), b22e.GetPtr(i-1), &mu, rwork.GetPtr(iu2cs+i-1-1), rwork.GetPtr(iu2sn+i-1-1))
			}
			rwork.Set(iu2cs+i-1-1, -rwork.Get(iu2cs+i-1-1))
			rwork.Set(iu2sn+i-1-1, -rwork.Get(iu2sn+i-1-1))

			temp = rwork.Get(iu1cs+i-1-1)*b11e.Get(i-1) + rwork.Get(iu1sn+i-1-1)*b11d.Get(i)
			b11d.Set(i, rwork.Get(iu1cs+i-1-1)*b11d.Get(i)-rwork.Get(iu1sn+i-1-1)*b11e.Get(i-1))
			b11e.Set(i-1, temp)
			if i < imax-1 {
				b11bulge = rwork.Get(iu1sn+i-1-1) * b11e.Get(i)
				b11e.Set(i, rwork.Get(iu1cs+i-1-1)*b11e.Get(i))
			}
			temp = rwork.Get(iu2cs+i-1-1)*b21e.Get(i-1) + rwork.Get(iu2sn+i-1-1)*b21d.Get(i)
			b21d.Set(i, rwork.Get(iu2cs+i-1-1)*b21d.Get(i)-rwork.Get(iu2sn+i-1-1)*b21e.Get(i-1))
			b21e.Set(i-1, temp)
			if i < imax-1 {
				b21bulge = rwork.Get(iu2sn+i-1-1) * b21e.Get(i)
				b21e.Set(i, rwork.Get(iu2cs+i-1-1)*b21e.Get(i))
			}
			temp = rwork.Get(iu1cs+i-1-1)*b12d.Get(i-1) + rwork.Get(iu1sn+i-1-1)*b12e.Get(i-1)
			b12e.Set(i-1, rwork.Get(iu1cs+i-1-1)*b12e.Get(i-1)-rwork.Get(iu1sn+i-1-1)*b12d.Get(i-1))
			b12d.Set(i-1, temp)
			b12bulge = rwork.Get(iu1sn+i-1-1) * b12d.Get(i)
			b12d.Set(i, rwork.Get(iu1cs+i-1-1)*b12d.Get(i))
			temp = rwork.Get(iu2cs+i-1-1)*b22d.Get(i-1) + rwork.Get(iu2sn+i-1-1)*b22e.Get(i-1)
			b22e.Set(i-1, rwork.Get(iu2cs+i-1-1)*b22e.Get(i-1)-rwork.Get(iu2sn+i-1-1)*b22d.Get(i-1))
			b22d.Set(i-1, temp)
			b22bulge = rwork.Get(iu2sn+i-1-1) * b22d.Get(i)
			b22d.Set(i, rwork.Get(iu2cs+i-1-1)*b22d.Get(i))

		}

		//        Compute PHI(IMAX-1)
		x1 = math.Sin(theta.Get(imax-1-1))*b11e.Get(imax-1-1) + math.Cos(theta.Get(imax-1-1))*b21e.Get(imax-1-1)
		y1 = math.Sin(theta.Get(imax-1-1))*b12d.Get(imax-1-1) + math.Cos(theta.Get(imax-1-1))*b22d.Get(imax-1-1)
		y2 = math.Sin(theta.Get(imax-1-1))*b12bulge + math.Cos(theta.Get(imax-1-1))*b22bulge

		phi.Set(imax-1-1, math.Atan2(math.Abs(x1), math.Sqrt(math.Pow(y1, 2)+math.Pow(y2, 2))))

		//        Chase bulges from B12(IMAX-1,IMAX) and B22(IMAX-1,IMAX)
		restart12 = math.Pow(b12d.Get(imax-1-1), 2)+math.Pow(b12bulge, 2) <= math.Pow(thresh, 2)
		restart22 = math.Pow(b22d.Get(imax-1-1), 2)+math.Pow(b22bulge, 2) <= math.Pow(thresh, 2)

		if !restart12 && !restart22 {
			Dlartgp(&y2, &y1, rwork.GetPtr(iv2tsn+imax-1-1-1), rwork.GetPtr(iv2tcs+imax-1-1-1), &r)
		} else if !restart12 && restart22 {
			Dlartgp(&b12bulge, b12d.GetPtr(imax-1-1), rwork.GetPtr(iv2tsn+imax-1-1-1), rwork.GetPtr(iv2tcs+imax-1-1-1), &r)
		} else if restart12 && !restart22 {
			Dlartgp(&b22bulge, b22d.GetPtr(imax-1-1), rwork.GetPtr(iv2tsn+imax-1-1-1), rwork.GetPtr(iv2tcs+imax-1-1-1), &r)
		} else if nu < mu {
			Dlartgs(b12e.GetPtr(imax-1-1), b12d.GetPtr(imax-1), &nu, rwork.GetPtr(iv2tcs+imax-1-1-1), rwork.GetPtr(iv2tsn+imax-1-1-1))
		} else {
			Dlartgs(b22e.GetPtr(imax-1-1), b22d.GetPtr(imax-1), &mu, rwork.GetPtr(iv2tcs+imax-1-1-1), rwork.GetPtr(iv2tsn+imax-1-1-1))
		}

		temp = rwork.Get(iv2tcs+imax-1-1-1)*b12e.Get(imax-1-1) + rwork.Get(iv2tsn+imax-1-1-1)*b12d.Get(imax-1)
		b12d.Set(imax-1, rwork.Get(iv2tcs+imax-1-1-1)*b12d.Get(imax-1)-rwork.Get(iv2tsn+imax-1-1-1)*b12e.Get(imax-1-1))
		b12e.Set(imax-1-1, temp)
		temp = rwork.Get(iv2tcs+imax-1-1-1)*b22e.Get(imax-1-1) + rwork.Get(iv2tsn+imax-1-1-1)*b22d.Get(imax-1)
		b22d.Set(imax-1, rwork.Get(iv2tcs+imax-1-1-1)*b22d.Get(imax-1)-rwork.Get(iv2tsn+imax-1-1-1)*b22e.Get(imax-1-1))
		b22e.Set(imax-1-1, temp)

		//        Update singular vectors
		if wantu1 {
			if colmajor {
				Zlasr('R', 'V', 'F', p, toPtr(imax-imin+1), rwork.Off(iu1cs+imin-1-1), rwork.Off(iu1sn+imin-1-1), u1.Off(0, imin-1), ldu1)
			} else {
				Zlasr('L', 'V', 'F', toPtr(imax-imin+1), p, rwork.Off(iu1cs+imin-1-1), rwork.Off(iu1sn+imin-1-1), u1.Off(imin-1, 0), ldu1)
			}
		}
		if wantu2 {
			if colmajor {
				Zlasr('R', 'V', 'F', toPtr((*m)-(*p)), toPtr(imax-imin+1), rwork.Off(iu2cs+imin-1-1), rwork.Off(iu2sn+imin-1-1), u2.Off(0, imin-1), ldu2)
			} else {
				Zlasr('L', 'V', 'F', toPtr(imax-imin+1), toPtr((*m)-(*p)), rwork.Off(iu2cs+imin-1-1), rwork.Off(iu2sn+imin-1-1), u2.Off(imin-1, 0), ldu2)
			}
		}
		if wantv1t {
			if colmajor {
				Zlasr('L', 'V', 'F', toPtr(imax-imin+1), q, rwork.Off(iv1tcs+imin-1-1), rwork.Off(iv1tsn+imin-1-1), v1t.Off(imin-1, 0), ldv1t)
			} else {
				Zlasr('R', 'V', 'F', q, toPtr(imax-imin+1), rwork.Off(iv1tcs+imin-1-1), rwork.Off(iv1tsn+imin-1-1), v1t.Off(0, imin-1), ldv1t)
			}
		}
		if wantv2t {
			if colmajor {
				Zlasr('L', 'V', 'F', toPtr(imax-imin+1), toPtr((*m)-(*q)), rwork.Off(iv2tcs+imin-1-1), rwork.Off(iv2tsn+imin-1-1), v2t.Off(imin-1, 0), ldv2t)
			} else {
				Zlasr('R', 'V', 'F', toPtr((*m)-(*q)), toPtr(imax-imin+1), rwork.Off(iv2tcs+imin-1-1), rwork.Off(iv2tsn+imin-1-1), v2t.Off(0, imin-1), ldv2t)
			}
		}

		//        Fix signs on B11(IMAX-1,IMAX) and B21(IMAX-1,IMAX)
		if b11e.Get(imax-1-1)+b21e.Get(imax-1-1) > 0 {
			b11d.Set(imax-1, -b11d.Get(imax-1))
			b21d.Set(imax-1, -b21d.Get(imax-1))
			if wantv1t {
				if colmajor {
					goblas.Zscal(*q, negonecomplex, v1t.CVector(imax-1, 0, *ldv1t))
				} else {
					goblas.Zscal(*q, negonecomplex, v1t.CVector(0, imax-1, 1))
				}
			}
		}

		//        Compute THETA(IMAX)
		x1 = math.Cos(phi.Get(imax-1-1))*b11d.Get(imax-1) + math.Sin(phi.Get(imax-1-1))*b12e.Get(imax-1-1)
		y1 = math.Cos(phi.Get(imax-1-1))*b21d.Get(imax-1) + math.Sin(phi.Get(imax-1-1))*b22e.Get(imax-1-1)

		theta.Set(imax-1, math.Atan2(math.Abs(y1), math.Abs(x1)))

		//        Fix signs on B11(IMAX,IMAX), B12(IMAX,IMAX-1), B21(IMAX,IMAX),
		//        and B22(IMAX,IMAX-1)
		if b11d.Get(imax-1)+b12e.Get(imax-1-1) < 0 {
			b12d.Set(imax-1, -b12d.Get(imax-1))
			if wantu1 {
				if colmajor {
					goblas.Zscal(*p, negonecomplex, u1.CVector(0, imax-1, 1))
				} else {
					goblas.Zscal(*p, negonecomplex, u1.CVector(imax-1, 0, *ldu1))
				}
			}
		}
		if b21d.Get(imax-1)+b22e.Get(imax-1-1) > 0 {
			b22d.Set(imax-1, -b22d.Get(imax-1))
			if wantu2 {
				if colmajor {
					goblas.Zscal((*m)-(*p), negonecomplex, u2.CVector(0, imax-1, 1))
				} else {
					goblas.Zscal((*m)-(*p), negonecomplex, u2.CVector(imax-1, 0, *ldu2))
				}
			}
		}

		//        Fix signs on B12(IMAX,IMAX) and B22(IMAX,IMAX)
		if b12d.Get(imax-1)+b22d.Get(imax-1) < 0 {
			if wantv2t {
				if colmajor {
					goblas.Zscal((*m)-(*q), negonecomplex, v2t.CVector(imax-1, 0, *ldv2t))
				} else {
					goblas.Zscal((*m)-(*q), negonecomplex, v2t.CVector(0, imax-1, 1))
				}
			}
		}

		//        Test for negligible sines or cosines
		for i = imin; i <= imax; i++ {
			if theta.Get(i-1) < thresh {
				theta.Set(i-1, zero)
			} else if theta.Get(i-1) > piover2-thresh {
				theta.Set(i-1, piover2)
			}
		}
		for i = imin; i <= imax-1; i++ {
			if phi.Get(i-1) < thresh {
				phi.Set(i-1, zero)
			} else if phi.Get(i-1) > piover2-thresh {
				phi.Set(i-1, piover2)
			}
		}

		//        Deflate
		if imax > 1 {
			for phi.Get(imax-1-1) == zero {
				imax = imax - 1
				if imax <= 1 {
					break
				}
			}
		}
		if imin > imax-1 {
			imin = imax - 1
		}
		if imin > 1 {
			for phi.Get(imin-1-1) != zero {
				imin = imin - 1
				if imin <= 1 {
					break
				}
			}
		}

		//        Repeat main iteration loop
	}

	//     Postprocessing: order THETA from least to greatest
	for i = 1; i <= (*q); i++ {

		mini = i
		thetamin = theta.Get(i - 1)
		for j = i + 1; j <= (*q); j++ {
			if theta.Get(j-1) < thetamin {
				mini = j
				thetamin = theta.Get(j - 1)
			}
		}

		if mini != i {
			theta.Set(mini-1, theta.Get(i-1))
			theta.Set(i-1, thetamin)
			if colmajor {
				if wantu1 {
					goblas.Zswap(*p, u1.CVector(0, i-1, 1), u1.CVector(0, mini-1, 1))
				}
				if wantu2 {
					goblas.Zswap((*m)-(*p), u2.CVector(0, i-1, 1), u2.CVector(0, mini-1, 1))
				}
				if wantv1t {
					goblas.Zswap(*q, v1t.CVector(i-1, 0, *ldv1t), v1t.CVector(mini-1, 0, *ldv1t))
				}
				if wantv2t {
					goblas.Zswap((*m)-(*q), v2t.CVector(i-1, 0, *ldv2t), v2t.CVector(mini-1, 0, *ldv2t))
				}
			} else {
				if wantu1 {
					goblas.Zswap(*p, u1.CVector(i-1, 0, *ldu1), u1.CVector(mini-1, 0, *ldu1))
				}
				if wantu2 {
					goblas.Zswap((*m)-(*p), u2.CVector(i-1, 0, *ldu2), u2.CVector(mini-1, 0, *ldu2))
				}
				if wantv1t {
					goblas.Zswap(*q, v1t.CVector(0, i-1, 1), v1t.CVector(0, mini-1, 1))
				}
				if wantv2t {
					goblas.Zswap((*m)-(*q), v2t.CVector(0, i-1, 1), v2t.CVector(0, mini-1, 1))
				}
			}
		}

	}
}
