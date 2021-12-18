package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dbbcsd  computes the CS decomposition of an orthogonal matrix in
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
//                   [ U1 |    ] [  0 |  0 -I  0 ] [ V1 |    ]**T
//                 = [---------] [---------------] [---------]   .
//                   [    | U2 ] [  S |  C  0  0 ] [    | V2 ]
//                               [  0 |  0  0  I ]
//
// X is M-by-M, its top-left block is P-by-Q, and Q must be no larger
// than P, M-P, or M-Q. (If Q is not the smallest index, then X must be
// transposed and/or permuted. This can be done in constant time using
// the TRANS and SIGNS options. See DORCSD for details.)
//
// The bidiagonal matrices B11, B12, B21, and B22 are represented
// implicitly by angles THETA(1:Q) and PHI(1:Q-1).
//
// The orthogonal matrices U1, U2, V1T, and V2T are input/output.
// The input matrices are pre- or post-multiplied by the appropriate
// singular vector matrices.
func Dbbcsd(jobu1, jobu2, jobv1t, jobv2t byte, trans mat.MatTrans, m, p, q int, theta, phi *mat.Vector, u1, u2, v1t, v2t *mat.Matrix, b11d, b11e, b12d, b12e, b21d, b21e, b22d, b22e, work *mat.Vector, lwork int) (info int, err error) {
	var colmajor, lquery, restart11, restart12, restart21, restart22, wantu1, wantu2, wantv1t, wantv2t bool
	var b11bulge, b12bulge, b21bulge, b22bulge, eps, hundred, meighth, mu, negone, nu, one, piover2, sigma11, sigma21, temp, ten, thetamax, thetamin, thresh, tol, tolmul, unfl, x1, x2, y1, y2, zero float64
	var i, imax, imin, iter, iu1cs, iu1sn, iu2cs, iu2sn, iv1tcs, iv1tsn, iv2tcs, iv2tsn, j, lworkmin, lworkopt, maxit, maxitr, mini int

	maxitr = 6
	hundred = 100.0
	meighth = -0.125
	one = 1.0
	piover2 = 1.57079632679489662
	ten = 10.0
	zero = 0.0
	negone = -1.0

	//     Test input arguments
	lquery = lwork == -1
	wantu1 = jobu1 == 'Y'
	wantu2 = jobu2 == 'Y'
	wantv1t = jobv1t == 'Y'
	wantv2t = jobv2t == 'Y'
	colmajor = trans != Trans

	if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if p < 0 || p > m {
		err = fmt.Errorf("p < 0 || p > m: m=%v, p=%v", m, p)
	} else if q < 0 || q > m {
		err = fmt.Errorf("q < 0 || q > m: m=%v, q=%v", m, q)
	} else if q > p || q > m-p || q > m-q {
		err = fmt.Errorf("q > p || q > m-p || q > m-q: m=%v, p=%v, q=%v", m, p, q)
	} else if wantu1 && u1.Rows < p {
		err = fmt.Errorf("wantu1 && u1.Rows < p: jobu1='%c', u1.Rows=%v, p=%v", jobu1, u1.Rows, p)
	} else if wantu2 && u2.Rows < m-p {
		err = fmt.Errorf("wantu2 && u2.Rows < m-p: jobu2='%c', u2.Rows=%v, m=%v, p=%v", jobu2, u2.Rows, m, p)
	} else if wantv1t && v1t.Rows < q {
		err = fmt.Errorf("wantv1t && v1t.Rows < q: jobv1t='%c', v1t.Rows=%v, q=%v", jobv1t, v1t.Rows, q)
	} else if wantv2t && v2t.Rows < m-q {
		err = fmt.Errorf("wantv2t && v2t.Rows < m-q: jobv2t='%c', v2t.Rows=%v, m=%v, q=%v", jobv2t, v2t.Rows, m, q)
	}

	//     Quick return if Q = 0
	if err == nil && q == 0 {
		lworkmin = 1
		work.Set(0, float64(lworkmin))
		return
	}

	//     Compute workspace
	if err == nil {
		iu1cs = 1
		iu1sn = iu1cs + q
		iu2cs = iu1sn + q
		iu2sn = iu2cs + q
		iv1tcs = iu2sn + q
		iv1tsn = iv1tcs + q
		iv2tcs = iv1tsn + q
		iv2tsn = iv2tcs + q
		lworkopt = iv2tsn + q - 1
		lworkmin = lworkopt
		work.Set(0, float64(lworkopt))
		if lwork < lworkmin && !lquery {
			err = fmt.Errorf("lwork < lworkmin && !lquery: lwork=%v, lworkmin=%v, lquery=%v", lwork, lworkmin, lquery)
		}
	}

	if err != nil {
		gltest.Xerbla2("Dbbcsd", err)
		return
	} else if lquery {
		return
	}

	//     Get machine constants
	eps = Dlamch(Epsilon)
	unfl = Dlamch(SafeMinimum)
	tolmul = math.Max(ten, math.Min(hundred, math.Pow(eps, meighth)))
	tol = tolmul * eps
	thresh = math.Max(tol, float64(maxitr*q*q)*unfl)

	//     Test for negligible sines or cosines
	for i = 1; i <= q; i++ {
		if theta.Get(i-1) < thresh {
			theta.Set(i-1, zero)
		} else if theta.Get(i-1) > piover2-thresh {
			theta.Set(i-1, piover2)
		}
	}
	for i = 1; i <= q-1; i++ {
		if phi.Get(i-1) < thresh {
			phi.Set(i-1, zero)
		} else if phi.Get(i-1) > piover2-thresh {
			phi.Set(i-1, piover2)
		}
	}

	//     Initial deflation
	imax = q
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
	maxit = maxitr * q * q
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
			info = 0
			for i = 1; i <= q; i++ {
				if phi.Get(i-1) != zero {
					info++
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
			sigma11, _ = Dlas2(b11d.Get(imax-1-1), b11e.Get(imax-1-1), b11d.Get(imax-1))
			sigma21, _ = Dlas2(b21d.Get(imax-1-1), b21e.Get(imax-1-1), b21d.Get(imax-1))

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
			*work.GetPtr(iv1tcs + imin - 1 - 1), *work.GetPtr(iv1tsn + imin - 1 - 1) = Dlartgs(b11d.Get(imin-1), b11e.Get(imin-1), mu)
		} else {
			*work.GetPtr(iv1tcs + imin - 1 - 1), *work.GetPtr(iv1tsn + imin - 1 - 1) = Dlartgs(b21d.Get(imin-1), b21e.Get(imin-1), nu)
		}

		temp = work.Get(iv1tcs+imin-1-1)*b11d.Get(imin-1) + work.Get(iv1tsn+imin-1-1)*b11e.Get(imin-1)
		b11e.Set(imin-1, work.Get(iv1tcs+imin-1-1)*b11e.Get(imin-1)-work.Get(iv1tsn+imin-1-1)*b11d.Get(imin-1))
		b11d.Set(imin-1, temp)
		b11bulge = work.Get(iv1tsn+imin-1-1) * b11d.Get(imin)
		b11d.Set(imin, work.Get(iv1tcs+imin-1-1)*b11d.Get(imin))
		temp = work.Get(iv1tcs+imin-1-1)*b21d.Get(imin-1) + work.Get(iv1tsn+imin-1-1)*b21e.Get(imin-1)
		b21e.Set(imin-1, work.Get(iv1tcs+imin-1-1)*b21e.Get(imin-1)-work.Get(iv1tsn+imin-1-1)*b21d.Get(imin-1))
		b21d.Set(imin-1, temp)
		b21bulge = work.Get(iv1tsn+imin-1-1) * b21d.Get(imin)
		b21d.Set(imin, work.Get(iv1tcs+imin-1-1)*b21d.Get(imin))

		//        Compute THETA(IMIN)
		theta.Set(imin-1, math.Atan2(math.Sqrt(math.Pow(b21d.Get(imin-1), 2)+math.Pow(b21bulge, 2)), math.Sqrt(math.Pow(b11d.Get(imin-1), 2)+math.Pow(b11bulge, 2))))

		//        Chase the bulges in B11(IMIN+1,IMIN) and B21(IMIN+1,IMIN)
		if math.Pow(b11d.Get(imin-1), 2)+math.Pow(b11bulge, 2) > math.Pow(thresh, 2) {
			*work.GetPtr(iu1sn + imin - 1 - 1), *work.GetPtr(iu1cs + imin - 1 - 1), _ = Dlartgp(b11bulge, b11d.Get(imin-1))
		} else if mu <= nu {
			*work.GetPtr(iu1cs + imin - 1 - 1), *work.GetPtr(iu1sn + imin - 1 - 1) = Dlartgs(b11e.Get(imin-1), b11d.Get(imin), mu)
		} else {
			*work.GetPtr(iu1cs + imin - 1 - 1), *work.GetPtr(iu1sn + imin - 1 - 1) = Dlartgs(b12d.Get(imin-1), b12e.Get(imin-1), nu)
		}
		if math.Pow(b21d.Get(imin-1), 2)+math.Pow(b21bulge, 2) > math.Pow(thresh, 2) {
			*work.GetPtr(iu2sn + imin - 1 - 1), *work.GetPtr(iu2cs + imin - 1 - 1), _ = Dlartgp(b21bulge, b21d.Get(imin-1))
		} else if nu < mu {
			*work.GetPtr(iu2cs + imin - 1 - 1), *work.GetPtr(iu2sn + imin - 1 - 1) = Dlartgs(b21e.Get(imin-1), b21d.Get(imin), nu)
		} else {
			*work.GetPtr(iu2cs + imin - 1 - 1), *work.GetPtr(iu2sn + imin - 1 - 1) = Dlartgs(b22d.Get(imin-1), b22e.Get(imin-1), mu)
		}
		work.Set(iu2cs+imin-1-1, -work.Get(iu2cs+imin-1-1))
		work.Set(iu2sn+imin-1-1, -work.Get(iu2sn+imin-1-1))

		temp = work.Get(iu1cs+imin-1-1)*b11e.Get(imin-1) + work.Get(iu1sn+imin-1-1)*b11d.Get(imin)
		b11d.Set(imin, work.Get(iu1cs+imin-1-1)*b11d.Get(imin)-work.Get(iu1sn+imin-1-1)*b11e.Get(imin-1))
		b11e.Set(imin-1, temp)
		if imax > imin+1 {
			b11bulge = work.Get(iu1sn+imin-1-1) * b11e.Get(imin)
			b11e.Set(imin, work.Get(iu1cs+imin-1-1)*b11e.Get(imin))
		}
		temp = work.Get(iu1cs+imin-1-1)*b12d.Get(imin-1) + work.Get(iu1sn+imin-1-1)*b12e.Get(imin-1)
		b12e.Set(imin-1, work.Get(iu1cs+imin-1-1)*b12e.Get(imin-1)-work.Get(iu1sn+imin-1-1)*b12d.Get(imin-1))
		b12d.Set(imin-1, temp)
		b12bulge = work.Get(iu1sn+imin-1-1) * b12d.Get(imin)
		b12d.Set(imin, work.Get(iu1cs+imin-1-1)*b12d.Get(imin))
		temp = work.Get(iu2cs+imin-1-1)*b21e.Get(imin-1) + work.Get(iu2sn+imin-1-1)*b21d.Get(imin)
		b21d.Set(imin, work.Get(iu2cs+imin-1-1)*b21d.Get(imin)-work.Get(iu2sn+imin-1-1)*b21e.Get(imin-1))
		b21e.Set(imin-1, temp)
		if imax > imin+1 {
			b21bulge = work.Get(iu2sn+imin-1-1) * b21e.Get(imin)
			b21e.Set(imin, work.Get(iu2cs+imin-1-1)*b21e.Get(imin))
		}
		temp = work.Get(iu2cs+imin-1-1)*b22d.Get(imin-1) + work.Get(iu2sn+imin-1-1)*b22e.Get(imin-1)
		b22e.Set(imin-1, work.Get(iu2cs+imin-1-1)*b22e.Get(imin-1)-work.Get(iu2sn+imin-1-1)*b22d.Get(imin-1))
		b22d.Set(imin-1, temp)
		b22bulge = work.Get(iu2sn+imin-1-1) * b22d.Get(imin)
		b22d.Set(imin, work.Get(iu2cs+imin-1-1)*b22d.Get(imin))

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
				*work.GetPtr(iv1tsn + i - 1 - 1), *work.GetPtr(iv1tcs + i - 1 - 1), _ = Dlartgp(x2, x1)
			} else if !restart11 && restart21 {
				*work.GetPtr(iv1tsn + i - 1 - 1), *work.GetPtr(iv1tcs + i - 1 - 1), _ = Dlartgp(b11bulge, b11e.Get(i-1-1))
			} else if restart11 && !restart21 {
				*work.GetPtr(iv1tsn + i - 1 - 1), *work.GetPtr(iv1tcs + i - 1 - 1), _ = Dlartgp(b21bulge, b21e.Get(i-1-1))
			} else if mu <= nu {
				*work.GetPtr(iv1tcs + i - 1 - 1), *work.GetPtr(iv1tsn + i - 1 - 1) = Dlartgs(b11d.Get(i-1), b11e.Get(i-1), mu)
			} else {
				*work.GetPtr(iv1tcs + i - 1 - 1), *work.GetPtr(iv1tsn + i - 1 - 1) = Dlartgs(b21d.Get(i-1), b21e.Get(i-1), nu)
			}
			work.Set(iv1tcs+i-1-1, -work.Get(iv1tcs+i-1-1))
			work.Set(iv1tsn+i-1-1, -work.Get(iv1tsn+i-1-1))
			if !restart12 && !restart22 {
				*work.GetPtr(iv2tsn + i - 1 - 1 - 1), *work.GetPtr(iv2tcs + i - 1 - 1 - 1), _ = Dlartgp(y2, y1)
			} else if !restart12 && restart22 {
				*work.GetPtr(iv2tsn + i - 1 - 1 - 1), *work.GetPtr(iv2tcs + i - 1 - 1 - 1), _ = Dlartgp(b12bulge, b12d.Get(i-1-1))
			} else if restart12 && !restart22 {
				*work.GetPtr(iv2tsn + i - 1 - 1 - 1), *work.GetPtr(iv2tcs + i - 1 - 1 - 1), _ = Dlartgp(b22bulge, b22d.Get(i-1-1))
			} else if nu < mu {
				*work.GetPtr(iv2tcs + i - 1 - 1 - 1), *work.GetPtr(iv2tsn + i - 1 - 1 - 1) = Dlartgs(b12e.Get(i-1-1), b12d.Get(i-1), nu)
			} else {
				*work.GetPtr(iv2tcs + i - 1 - 1 - 1), *work.GetPtr(iv2tsn + i - 1 - 1 - 1) = Dlartgs(b22e.Get(i-1-1), b22d.Get(i-1), mu)
			}

			temp = work.Get(iv1tcs+i-1-1)*b11d.Get(i-1) + work.Get(iv1tsn+i-1-1)*b11e.Get(i-1)
			b11e.Set(i-1, work.Get(iv1tcs+i-1-1)*b11e.Get(i-1)-work.Get(iv1tsn+i-1-1)*b11d.Get(i-1))
			b11d.Set(i-1, temp)
			b11bulge = work.Get(iv1tsn+i-1-1) * b11d.Get(i)
			b11d.Set(i, work.Get(iv1tcs+i-1-1)*b11d.Get(i))
			temp = work.Get(iv1tcs+i-1-1)*b21d.Get(i-1) + work.Get(iv1tsn+i-1-1)*b21e.Get(i-1)
			b21e.Set(i-1, work.Get(iv1tcs+i-1-1)*b21e.Get(i-1)-work.Get(iv1tsn+i-1-1)*b21d.Get(i-1))
			b21d.Set(i-1, temp)
			b21bulge = work.Get(iv1tsn+i-1-1) * b21d.Get(i)
			b21d.Set(i, work.Get(iv1tcs+i-1-1)*b21d.Get(i))
			temp = work.Get(iv2tcs+i-1-1-1)*b12e.Get(i-1-1) + work.Get(iv2tsn+i-1-1-1)*b12d.Get(i-1)
			b12d.Set(i-1, work.Get(iv2tcs+i-1-1-1)*b12d.Get(i-1)-work.Get(iv2tsn+i-1-1-1)*b12e.Get(i-1-1))
			b12e.Set(i-1-1, temp)
			b12bulge = work.Get(iv2tsn+i-1-1-1) * b12e.Get(i-1)
			b12e.Set(i-1, work.Get(iv2tcs+i-1-1-1)*b12e.Get(i-1))
			temp = work.Get(iv2tcs+i-1-1-1)*b22e.Get(i-1-1) + work.Get(iv2tsn+i-1-1-1)*b22d.Get(i-1)
			b22d.Set(i-1, work.Get(iv2tcs+i-1-1-1)*b22d.Get(i-1)-work.Get(iv2tsn+i-1-1-1)*b22e.Get(i-1-1))
			b22e.Set(i-1-1, temp)
			b22bulge = work.Get(iv2tsn+i-1-1-1) * b22e.Get(i-1)
			b22e.Set(i-1, work.Get(iv2tcs+i-1-1-1)*b22e.Get(i-1))

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
				*work.GetPtr(iu1sn + i - 1 - 1), *work.GetPtr(iu1cs + i - 1 - 1), _ = Dlartgp(x2, x1)
			} else if !restart11 && restart12 {
				*work.GetPtr(iu1sn + i - 1 - 1), *work.GetPtr(iu1cs + i - 1 - 1), _ = Dlartgp(b11bulge, b11d.Get(i-1))
			} else if restart11 && !restart12 {
				*work.GetPtr(iu1sn + i - 1 - 1), *work.GetPtr(iu1cs + i - 1 - 1), _ = Dlartgp(b12bulge, b12e.Get(i-1-1))
			} else if mu <= nu {
				*work.GetPtr(iu1cs + i - 1 - 1), *work.GetPtr(iu1sn + i - 1 - 1) = Dlartgs(b11e.Get(i-1), b11d.Get(i), mu)
			} else {
				*work.GetPtr(iu1cs + i - 1 - 1), *work.GetPtr(iu1sn + i - 1 - 1) = Dlartgs(b12d.Get(i-1), b12e.Get(i-1), nu)
			}
			if !restart21 && !restart22 {
				*work.GetPtr(iu2sn + i - 1 - 1), *work.GetPtr(iu2cs + i - 1 - 1), _ = Dlartgp(y2, y1)
			} else if !restart21 && restart22 {
				*work.GetPtr(iu2sn + i - 1 - 1), *work.GetPtr(iu2cs + i - 1 - 1), _ = Dlartgp(b21bulge, b21d.Get(i-1))
			} else if restart21 && !restart22 {
				*work.GetPtr(iu2sn + i - 1 - 1), *work.GetPtr(iu2cs + i - 1 - 1), _ = Dlartgp(b22bulge, b22e.Get(i-1-1))
			} else if nu < mu {
				*work.GetPtr(iu2cs + i - 1 - 1), *work.GetPtr(iu2sn + i - 1 - 1) = Dlartgs(b21e.Get(i-1), b21e.Get(i), nu)
			} else {
				*work.GetPtr(iu2cs + i - 1 - 1), *work.GetPtr(iu2sn + i - 1 - 1) = Dlartgs(b22d.Get(i-1), b22e.Get(i-1), mu)
			}
			work.Set(iu2cs+i-1-1, -work.Get(iu2cs+i-1-1))
			work.Set(iu2sn+i-1-1, -work.Get(iu2sn+i-1-1))

			temp = work.Get(iu1cs+i-1-1)*b11e.Get(i-1) + work.Get(iu1sn+i-1-1)*b11d.Get(i)
			b11d.Set(i, work.Get(iu1cs+i-1-1)*b11d.Get(i)-work.Get(iu1sn+i-1-1)*b11e.Get(i-1))
			b11e.Set(i-1, temp)
			if i < imax-1 {
				b11bulge = work.Get(iu1sn+i-1-1) * b11e.Get(i)
				b11e.Set(i, work.Get(iu1cs+i-1-1)*b11e.Get(i))
			}
			temp = work.Get(iu2cs+i-1-1)*b21e.Get(i-1) + work.Get(iu2sn+i-1-1)*b21d.Get(i)
			b21d.Set(i, work.Get(iu2cs+i-1-1)*b21d.Get(i)-work.Get(iu2sn+i-1-1)*b21e.Get(i-1))
			b21e.Set(i-1, temp)
			if i < imax-1 {
				b21bulge = work.Get(iu2sn+i-1-1) * b21e.Get(i)
				b21e.Set(i, work.Get(iu2cs+i-1-1)*b21e.Get(i))
			}
			temp = work.Get(iu1cs+i-1-1)*b12d.Get(i-1) + work.Get(iu1sn+i-1-1)*b12e.Get(i-1)
			b12e.Set(i-1, work.Get(iu1cs+i-1-1)*b12e.Get(i-1)-work.Get(iu1sn+i-1-1)*b12d.Get(i-1))
			b12d.Set(i-1, temp)
			b12bulge = work.Get(iu1sn+i-1-1) * b12d.Get(i)
			b12d.Set(i, work.Get(iu1cs+i-1-1)*b12d.Get(i))
			temp = work.Get(iu2cs+i-1-1)*b22d.Get(i-1) + work.Get(iu2sn+i-1-1)*b22e.Get(i-1)
			b22e.Set(i-1, work.Get(iu2cs+i-1-1)*b22e.Get(i-1)-work.Get(iu2sn+i-1-1)*b22d.Get(i-1))
			b22d.Set(i-1, temp)
			b22bulge = work.Get(iu2sn+i-1-1) * b22d.Get(i)
			b22d.Set(i, work.Get(iu2cs+i-1-1)*b22d.Get(i))

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
			*work.GetPtr(iv2tsn + imax - 1 - 1 - 1), *work.GetPtr(iv2tcs + imax - 1 - 1 - 1), _ = Dlartgp(y2, y1)
		} else if !restart12 && restart22 {
			*work.GetPtr(iv2tsn + imax - 1 - 1 - 1), *work.GetPtr(iv2tcs + imax - 1 - 1 - 1), _ = Dlartgp(b12bulge, b12d.Get(imax-1-1))
		} else if restart12 && !restart22 {
			*work.GetPtr(iv2tsn + imax - 1 - 1 - 1), *work.GetPtr(iv2tcs + imax - 1 - 1 - 1), _ = Dlartgp(b22bulge, b22d.Get(imax-1-1))
		} else if nu < mu {
			*work.GetPtr(iv2tcs + imax - 1 - 1 - 1), *work.GetPtr(iv2tsn + imax - 1 - 1 - 1) = Dlartgs(b12e.Get(imax-1-1), b12d.Get(imax-1), nu)
		} else {
			*work.GetPtr(iv2tcs + imax - 1 - 1 - 1), *work.GetPtr(iv2tsn + imax - 1 - 1 - 1) = Dlartgs(b22e.Get(imax-1-1), b22d.Get(imax-1), mu)
		}

		temp = work.Get(iv2tcs+imax-1-1-1)*b12e.Get(imax-1-1) + work.Get(iv2tsn+imax-1-1-1)*b12d.Get(imax-1)
		b12d.Set(imax-1, work.Get(iv2tcs+imax-1-1-1)*b12d.Get(imax-1)-work.Get(iv2tsn+imax-1-1-1)*b12e.Get(imax-1-1))
		b12e.Set(imax-1-1, temp)
		temp = work.Get(iv2tcs+imax-1-1-1)*b22e.Get(imax-1-1) + work.Get(iv2tsn+imax-1-1-1)*b22d.Get(imax-1)
		b22d.Set(imax-1, work.Get(iv2tcs+imax-1-1-1)*b22d.Get(imax-1)-work.Get(iv2tsn+imax-1-1-1)*b22e.Get(imax-1-1))
		b22e.Set(imax-1-1, temp)

		//        Update singular vectors
		if wantu1 {
			if colmajor {
				if err = Dlasr(Right, 'V', 'F', p, imax-imin+1, work.Off(iu1cs+imin-1-1), work.Off(iu1sn+imin-1-1), u1.Off(0, imin-1)); err != nil {
					panic(err)
				}
			} else {
				if err = Dlasr(Left, 'V', 'F', imax-imin+1, p, work.Off(iu1cs+imin-1-1), work.Off(iu1sn+imin-1-1), u1.Off(imin-1, 0)); err != nil {
					panic(err)
				}
			}
		}
		if wantu2 {
			if colmajor {
				if err = Dlasr(Right, 'V', 'F', m-p, imax-imin+1, work.Off(iu2cs+imin-1-1), work.Off(iu2sn+imin-1-1), u2.Off(0, imin-1)); err != nil {
					panic(err)
				}
			} else {
				if err = Dlasr(Left, 'V', 'F', imax-imin+1, m-p, work.Off(iu2cs+imin-1-1), work.Off(iu2sn+imin-1-1), u2.Off(imin-1, 0)); err != nil {
					panic(err)
				}
			}
		}
		if wantv1t {
			if colmajor {
				if err = Dlasr(Left, 'V', 'F', imax-imin+1, q, work.Off(iv1tcs+imin-1-1), work.Off(iv1tsn+imin-1-1), v1t.Off(imin-1, 0)); err != nil {
					panic(err)
				}
			} else {
				if err = Dlasr(Right, 'V', 'F', q, imax-imin+1, work.Off(iv1tcs+imin-1-1), work.Off(iv1tsn+imin-1-1), v1t.Off(0, imin-1)); err != nil {
					panic(err)
				}
			}
		}
		if wantv2t {
			if colmajor {
				if err = Dlasr(Left, 'V', 'F', imax-imin+1, m-q, work.Off(iv2tcs+imin-1-1), work.Off(iv2tsn+imin-1-1), v2t.Off(imin-1, 0)); err != nil {
					panic(err)
				}
			} else {
				if err = Dlasr(Right, 'V', 'F', m-q, imax-imin+1, work.Off(iv2tcs+imin-1-1), work.Off(iv2tsn+imin-1-1), v2t.Off(0, imin-1)); err != nil {
					panic(err)
				}
			}
		}

		//        Fix signs on B11(IMAX-1,IMAX) and B21(IMAX-1,IMAX)
		if b11e.Get(imax-1-1)+b21e.Get(imax-1-1) > 0 {
			b11d.Set(imax-1, -b11d.Get(imax-1))
			b21d.Set(imax-1, -b21d.Get(imax-1))
			if wantv1t {
				if colmajor {
					v1t.Off(imax-1, 0).Vector().Scal(q, negone, v1t.Rows)
				} else {
					v1t.Off(0, imax-1).Vector().Scal(q, negone, 1)
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
					u1.Off(0, imax-1).Vector().Scal(p, negone, 1)
				} else {
					u1.Off(imax-1, 0).Vector().Scal(p, negone, u1.Rows)
				}
			}
		}
		if b21d.Get(imax-1)+b22e.Get(imax-1-1) > 0 {
			b22d.Set(imax-1, -b22d.Get(imax-1))
			if wantu2 {
				if colmajor {
					u2.Off(0, imax-1).Vector().Scal(m-p, negone, 1)
				} else {
					u2.Off(imax-1, 0).Vector().Scal(m-p, negone, u2.Rows)
				}
			}
		}

		//        Fix signs on B12(IMAX,IMAX) and B22(IMAX,IMAX)
		if b12d.Get(imax-1)+b22d.Get(imax-1) < 0 {
			if wantv2t {
				if colmajor {
					v2t.Off(imax-1, 0).Vector().Scal(m-q, negone, v2t.Rows)
				} else {
					v2t.Off(0, imax-1).Vector().Scal(m-q, negone, 1)
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
	for i = 1; i <= q; i++ {

		mini = i
		thetamin = theta.Get(i - 1)
		for j = i + 1; j <= q; j++ {
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
					u1.Off(0, mini-1).Vector().Swap(p, u1.Off(0, i-1).Vector(), 1, 1)
				}
				if wantu2 {
					u2.Off(0, mini-1).Vector().Swap(m-p, u2.Off(0, i-1).Vector(), 1, 1)
				}
				if wantv1t {
					v1t.Off(mini-1, 0).Vector().Swap(q, v1t.Off(i-1, 0).Vector(), v1t.Rows, v1t.Rows)
				}
				if wantv2t {
					v2t.Off(mini-1, 0).Vector().Swap(m-q, v2t.Off(i-1, 0).Vector(), v2t.Rows, v2t.Rows)
				}
			} else {
				if wantu1 {
					u1.Off(mini-1, 0).Vector().Swap(p, u1.Off(i-1, 0).Vector(), u1.Rows, u1.Rows)
				}
				if wantu2 {
					u2.Off(mini-1, 0).Vector().Swap(m-p, u2.Off(i-1, 0).Vector(), u2.Rows, u2.Rows)
				}
				if wantv1t {
					v1t.Off(0, mini-1).Vector().Swap(q, v1t.Off(0, i-1).Vector(), 1, 1)
				}
				if wantv2t {
					v2t.Off(0, mini-1).Vector().Swap(m-q, v2t.Off(0, i-1).Vector(), 1, 1)
				}
			}
		}

	}

	return
}
