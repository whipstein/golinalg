package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Ztprfs provides error bounds and backward error estimates for the
// solution to a system of linear equations with a triangular packed
// coefficient matrix.
//
// The solution matrix X must be computed by ZTPTRS or some other
// means before entering this routine.  ZTPRFS does not do iterative
// refinement because doing so cannot improve the backward error.
func Ztprfs(uplo, trans, diag byte, n, nrhs *int, ap *mat.CVector, b *mat.CMatrix, ldb *int, x *mat.CMatrix, ldx *int, ferr, berr *mat.Vector, work *mat.CVector, rwork *mat.Vector, info *int) {
	var notran, nounit, upper bool
	var transn, transt byte
	var one complex128
	var eps, lstres, s, safe1, safe2, safmin, xk, zero float64
	var i, j, k, kase, kc, nz int
	isave := make([]int, 3)

	zero = 0.0
	one = (1.0 + 0.0*1i)

	Cabs1 := func(zdum complex128) float64 { return math.Abs(real(zdum)) + math.Abs(imag(zdum)) }

	//     Test the input parameters.
	(*info) = 0
	upper = uplo == 'U'
	notran = trans == 'N'
	nounit = diag == 'N'

	if !upper && uplo != 'L' {
		(*info) = -1
	} else if !notran && trans != 'T' && trans != 'C' {
		(*info) = -2
	} else if !nounit && diag != 'U' {
		(*info) = -3
	} else if (*n) < 0 {
		(*info) = -4
	} else if (*nrhs) < 0 {
		(*info) = -5
	} else if (*ldb) < maxint(1, *n) {
		(*info) = -8
	} else if (*ldx) < maxint(1, *n) {
		(*info) = -10
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZTPRFS"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 || (*nrhs) == 0 {
		for j = 1; j <= (*nrhs); j++ {
			ferr.Set(j-1, zero)
			berr.Set(j-1, zero)
		}
		return
	}

	if notran {
		transn = 'N'
		transt = 'C'
	} else {
		transn = 'C'
		transt = 'N'
	}

	//     NZ = maximum number of nonzero elements in each row of A, plus 1
	nz = (*n) + 1
	eps = Dlamch(Epsilon)
	safmin = Dlamch(SafeMinimum)
	safe1 = float64(nz) * safmin
	safe2 = safe1 / eps

	//     Do for each right hand side
	for j = 1; j <= (*nrhs); j++ {
		//        Compute residual R = B - op(A) * X,
		//
		goblas.Zcopy(n, x.CVector(0, j-1), func() *int { y := 1; return &y }(), work, func() *int { y := 1; return &y }())
		goblas.Ztpmv(mat.UploByte(uplo), mat.TransByte(trans), mat.DiagByte(diag), n, ap, work, func() *int { y := 1; return &y }())
		goblas.Zaxpy(n, toPtrc128(-one), b.CVector(0, j-1), func() *int { y := 1; return &y }(), work, func() *int { y := 1; return &y }())

		//        Compute componentwise relative backward error from formula
		//
		//        maxint(i) ( abs(R(i)) / ( abs(op(A))*abs(X) + abs(B) )(i) )
		//
		//        where abs(Z) is the componentwise absolute value of the matrix
		//        or vector Z.  If the i-th component of the denominator is less
		//        than SAFE2, then SAFE1 is added to the i-th components of the
		//        numerator and denominator before dividing.
		for i = 1; i <= (*n); i++ {
			rwork.Set(i-1, Cabs1(b.Get(i-1, j-1)))
		}

		if notran {
			//           Compute abs(A)*abs(X) + abs(B).
			if upper {
				kc = 1
				if nounit {
					for k = 1; k <= (*n); k++ {
						xk = Cabs1(x.Get(k-1, j-1))
						for i = 1; i <= k; i++ {
							rwork.Set(i-1, rwork.Get(i-1)+Cabs1(ap.Get(kc+i-1-1))*xk)
						}
						kc = kc + k
					}
				} else {
					for k = 1; k <= (*n); k++ {
						xk = Cabs1(x.Get(k-1, j-1))
						for i = 1; i <= k-1; i++ {
							rwork.Set(i-1, rwork.Get(i-1)+Cabs1(ap.Get(kc+i-1-1))*xk)
						}
						rwork.Set(k-1, rwork.Get(k-1)+xk)
						kc = kc + k
					}
				}
			} else {
				kc = 1
				if nounit {
					for k = 1; k <= (*n); k++ {
						xk = Cabs1(x.Get(k-1, j-1))
						for i = k; i <= (*n); i++ {
							rwork.Set(i-1, rwork.Get(i-1)+Cabs1(ap.Get(kc+i-k-1))*xk)
						}
						kc = kc + (*n) - k + 1
					}
				} else {
					for k = 1; k <= (*n); k++ {
						xk = Cabs1(x.Get(k-1, j-1))
						for i = k + 1; i <= (*n); i++ {
							rwork.Set(i-1, rwork.Get(i-1)+Cabs1(ap.Get(kc+i-k-1))*xk)
						}
						rwork.Set(k-1, rwork.Get(k-1)+xk)
						kc = kc + (*n) - k + 1
					}
				}
			}
		} else {
			//           Compute abs(A**H)*abs(X) + abs(B).
			if upper {
				kc = 1
				if nounit {
					for k = 1; k <= (*n); k++ {
						s = zero
						for i = 1; i <= k; i++ {
							s = s + Cabs1(ap.Get(kc+i-1-1))*Cabs1(x.Get(i-1, j-1))
						}
						rwork.Set(k-1, rwork.Get(k-1)+s)
						kc = kc + k
					}
				} else {
					for k = 1; k <= (*n); k++ {
						s = Cabs1(x.Get(k-1, j-1))
						for i = 1; i <= k-1; i++ {
							s = s + Cabs1(ap.Get(kc+i-1-1))*Cabs1(x.Get(i-1, j-1))
						}
						rwork.Set(k-1, rwork.Get(k-1)+s)
						kc = kc + k
					}
				}
			} else {
				kc = 1
				if nounit {
					for k = 1; k <= (*n); k++ {
						s = zero
						for i = k; i <= (*n); i++ {
							s = s + Cabs1(ap.Get(kc+i-k-1))*Cabs1(x.Get(i-1, j-1))
						}
						rwork.Set(k-1, rwork.Get(k-1)+s)
						kc = kc + (*n) - k + 1
					}
				} else {
					for k = 1; k <= (*n); k++ {
						s = Cabs1(x.Get(k-1, j-1))
						for i = k + 1; i <= (*n); i++ {
							s = s + Cabs1(ap.Get(kc+i-k-1))*Cabs1(x.Get(i-1, j-1))
						}
						rwork.Set(k-1, rwork.Get(k-1)+s)
						kc = kc + (*n) - k + 1
					}
				}
			}
		}
		s = zero
		for i = 1; i <= (*n); i++ {
			if rwork.Get(i-1) > safe2 {
				s = maxf64(s, Cabs1(work.Get(i-1))/rwork.Get(i-1))
			} else {
				s = maxf64(s, (Cabs1(work.Get(i-1))+safe1)/(rwork.Get(i-1)+safe1))
			}
		}
		berr.Set(j-1, s)

		//        Bound error from formula
		//
		//        norm(X - XTRUE) / norm(X) .le. FERR =
		//        norm( abs(inv(op(A)))*
		//           ( abs(R) + NZ*EPS*( abs(op(A))*abs(X)+abs(B) ))) / norm(X)
		//
		//        where
		//          norm(Z) is the magnitude of the largest component of Z
		//          inv(op(A)) is the inverse of op(A)
		//          abs(Z) is the componentwise absolute value of the matrix or
		//             vector Z
		//          NZ is the maximum number of nonzeros in any row of A, plus 1
		//          EPS is machine epsilon
		//
		//        The i-th component of abs(R)+NZ*EPS*(abs(op(A))*abs(X)+abs(B))
		//        is incremented by SAFE1 if the i-th component of
		//        abs(op(A))*abs(X) + abs(B) is less than SAFE2.
		//
		//        Use ZLACN2 to estimate the infinity-norm of the matrix
		//           inv(op(A)) * diag(W),
		//        where W = abs(R) + NZ*EPS*( abs(op(A))*abs(X)+abs(B) )))
		for i = 1; i <= (*n); i++ {
			if rwork.Get(i-1) > safe2 {
				rwork.Set(i-1, Cabs1(work.Get(i-1))+float64(nz)*eps*rwork.Get(i-1))
			} else {
				rwork.Set(i-1, Cabs1(work.Get(i-1))+float64(nz)*eps*rwork.Get(i-1)+safe1)
			}
		}

		kase = 0
	label210:
		;
		Zlacn2(n, work.Off((*n)+1-1), work, ferr.GetPtr(j-1), &kase, &isave)
		if kase != 0 {
			if kase == 1 {
				//              Multiply by diag(W)*inv(op(A)**H).
				goblas.Ztpsv(mat.UploByte(uplo), mat.TransByte(transt), mat.DiagByte(diag), n, ap, work, func() *int { y := 1; return &y }())
				for i = 1; i <= (*n); i++ {
					work.Set(i-1, rwork.GetCmplx(i-1)*work.Get(i-1))
				}
			} else {
				//              Multiply by inv(op(A))*diag(W).
				for i = 1; i <= (*n); i++ {
					work.Set(i-1, rwork.GetCmplx(i-1)*work.Get(i-1))
				}
				goblas.Ztpsv(mat.UploByte(uplo), mat.TransByte(transn), mat.DiagByte(diag), n, ap, work, func() *int { y := 1; return &y }())
			}
			goto label210
		}

		//        Normalize error.
		lstres = zero
		for i = 1; i <= (*n); i++ {
			lstres = maxf64(lstres, Cabs1(x.Get(i-1, j-1)))
		}
		if lstres != zero {
			ferr.Set(j-1, ferr.Get(j-1)/lstres)
		}

	}
}
