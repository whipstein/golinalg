package golapack

import (
	"fmt"
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
// means before entering this routine.  Ztprfs does not do iterative
// refinement because doing so cannot improve the backward error.
func Ztprfs(uplo mat.MatUplo, trans mat.MatTrans, diag mat.MatDiag, n, nrhs int, ap *mat.CVector, b, x *mat.CMatrix, ferr, berr *mat.Vector, work *mat.CVector, rwork *mat.Vector) (err error) {
	var notran, nounit, upper bool
	var transn, transt mat.MatTrans
	var one complex128
	var eps, lstres, s, safe1, safe2, safmin, xk, zero float64
	var i, j, k, kase, kc, nz int

	isave := make([]int, 3)

	zero = 0.0
	one = (1.0 + 0.0*1i)

	//     Test the input parameters.
	upper = uplo == Upper
	notran = trans == NoTrans
	nounit = diag == NonUnit

	if !upper && uplo != Lower {
		err = fmt.Errorf("!upper && uplo != Lower: uplo=%s", uplo)
	} else if !notran && trans != Trans && trans != ConjTrans {
		err = fmt.Errorf("!notran && trans != Trans && trans != ConjTrans: trans=%s", trans)
	} else if !nounit && diag != Unit {
		err = fmt.Errorf("!nounit && diag != Unit: diag=%s", diag)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if nrhs < 0 {
		err = fmt.Errorf("nrhs < 0: nrhs=%v", nrhs)
	} else if b.Rows < max(1, n) {
		err = fmt.Errorf("b.Rows < max(1, n): b.Rows=%v, n=%v", b.Rows, n)
	} else if x.Rows < max(1, n) {
		err = fmt.Errorf("x.Rows < max(1, n): x.Rows=%v, n=%v", x.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Ztprfs", err)
		return
	}

	//     Quick return if possible
	if n == 0 || nrhs == 0 {
		for j = 1; j <= nrhs; j++ {
			ferr.Set(j-1, zero)
			berr.Set(j-1, zero)
		}
		return
	}

	if notran {
		transn = NoTrans
		transt = ConjTrans
	} else {
		transn = ConjTrans
		transt = NoTrans
	}

	//     NZ = maximum number of nonzero elements in each row of A, plus 1
	nz = n + 1
	eps = Dlamch(Epsilon)
	safmin = Dlamch(SafeMinimum)
	safe1 = float64(nz) * safmin
	safe2 = safe1 / eps

	//     Do for each right hand side
	for j = 1; j <= nrhs; j++ {
		//        Compute residual R = B - op(A) * X,
		//
		goblas.Zcopy(n, x.CVector(0, j-1, 1), work.Off(0, 1))
		if err = goblas.Ztpmv(uplo, trans, diag, n, ap, work.Off(0, 1)); err != nil {
			panic(err)
		}
		goblas.Zaxpy(n, -one, b.CVector(0, j-1, 1), work.Off(0, 1))

		//        Compute componentwise relative backward error from formula
		//
		//        max(i) ( abs(R(i)) / ( abs(op(A))*abs(X) + abs(B) )(i) )
		//
		//        where abs(Z) is the componentwise absolute value of the matrix
		//        or vector Z.  If the i-th component of the denominator is less
		//        than SAFE2, then SAFE1 is added to the i-th components of the
		//        numerator and denominator before dividing.
		for i = 1; i <= n; i++ {
			rwork.Set(i-1, cabs1(b.Get(i-1, j-1)))
		}

		if notran {
			//           Compute abs(A)*abs(X) + abs(B).
			if upper {
				kc = 1
				if nounit {
					for k = 1; k <= n; k++ {
						xk = cabs1(x.Get(k-1, j-1))
						for i = 1; i <= k; i++ {
							rwork.Set(i-1, rwork.Get(i-1)+cabs1(ap.Get(kc+i-1-1))*xk)
						}
						kc = kc + k
					}
				} else {
					for k = 1; k <= n; k++ {
						xk = cabs1(x.Get(k-1, j-1))
						for i = 1; i <= k-1; i++ {
							rwork.Set(i-1, rwork.Get(i-1)+cabs1(ap.Get(kc+i-1-1))*xk)
						}
						rwork.Set(k-1, rwork.Get(k-1)+xk)
						kc = kc + k
					}
				}
			} else {
				kc = 1
				if nounit {
					for k = 1; k <= n; k++ {
						xk = cabs1(x.Get(k-1, j-1))
						for i = k; i <= n; i++ {
							rwork.Set(i-1, rwork.Get(i-1)+cabs1(ap.Get(kc+i-k-1))*xk)
						}
						kc = kc + n - k + 1
					}
				} else {
					for k = 1; k <= n; k++ {
						xk = cabs1(x.Get(k-1, j-1))
						for i = k + 1; i <= n; i++ {
							rwork.Set(i-1, rwork.Get(i-1)+cabs1(ap.Get(kc+i-k-1))*xk)
						}
						rwork.Set(k-1, rwork.Get(k-1)+xk)
						kc = kc + n - k + 1
					}
				}
			}
		} else {
			//           Compute abs(A**H)*abs(X) + abs(B).
			if upper {
				kc = 1
				if nounit {
					for k = 1; k <= n; k++ {
						s = zero
						for i = 1; i <= k; i++ {
							s = s + cabs1(ap.Get(kc+i-1-1))*cabs1(x.Get(i-1, j-1))
						}
						rwork.Set(k-1, rwork.Get(k-1)+s)
						kc = kc + k
					}
				} else {
					for k = 1; k <= n; k++ {
						s = cabs1(x.Get(k-1, j-1))
						for i = 1; i <= k-1; i++ {
							s = s + cabs1(ap.Get(kc+i-1-1))*cabs1(x.Get(i-1, j-1))
						}
						rwork.Set(k-1, rwork.Get(k-1)+s)
						kc = kc + k
					}
				}
			} else {
				kc = 1
				if nounit {
					for k = 1; k <= n; k++ {
						s = zero
						for i = k; i <= n; i++ {
							s = s + cabs1(ap.Get(kc+i-k-1))*cabs1(x.Get(i-1, j-1))
						}
						rwork.Set(k-1, rwork.Get(k-1)+s)
						kc = kc + n - k + 1
					}
				} else {
					for k = 1; k <= n; k++ {
						s = cabs1(x.Get(k-1, j-1))
						for i = k + 1; i <= n; i++ {
							s = s + cabs1(ap.Get(kc+i-k-1))*cabs1(x.Get(i-1, j-1))
						}
						rwork.Set(k-1, rwork.Get(k-1)+s)
						kc = kc + n - k + 1
					}
				}
			}
		}
		s = zero
		for i = 1; i <= n; i++ {
			if rwork.Get(i-1) > safe2 {
				s = math.Max(s, cabs1(work.Get(i-1))/rwork.Get(i-1))
			} else {
				s = math.Max(s, (cabs1(work.Get(i-1))+safe1)/(rwork.Get(i-1)+safe1))
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
		for i = 1; i <= n; i++ {
			if rwork.Get(i-1) > safe2 {
				rwork.Set(i-1, cabs1(work.Get(i-1))+float64(nz)*eps*rwork.Get(i-1))
			} else {
				rwork.Set(i-1, cabs1(work.Get(i-1))+float64(nz)*eps*rwork.Get(i-1)+safe1)
			}
		}

		kase = 0
	label210:
		;
		*ferr.GetPtr(j - 1), kase = Zlacn2(n, work.Off(n), work, ferr.Get(j-1), kase, &isave)
		if kase != 0 {
			if kase == 1 {
				//              Multiply by diag(W)*inv(op(A)**H).
				if err = goblas.Ztpsv(uplo, transt, diag, n, ap, work.Off(0, 1)); err != nil {
					panic(err)
				}
				for i = 1; i <= n; i++ {
					work.Set(i-1, rwork.GetCmplx(i-1)*work.Get(i-1))
				}
			} else {
				//              Multiply by inv(op(A))*diag(W).
				for i = 1; i <= n; i++ {
					work.Set(i-1, rwork.GetCmplx(i-1)*work.Get(i-1))
				}
				if err = goblas.Ztpsv(uplo, transn, diag, n, ap, work.Off(0, 1)); err != nil {
					panic(err)
				}
			}
			goto label210
		}

		//        Normalize error.
		lstres = zero
		for i = 1; i <= n; i++ {
			lstres = math.Max(lstres, cabs1(x.Get(i-1, j-1)))
		}
		if lstres != zero {
			ferr.Set(j-1, ferr.Get(j-1)/lstres)
		}

	}

	return
}
