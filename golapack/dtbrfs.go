package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dtbrfs provides error bounds and backward error estimates for the
// solution to a system of linear equations with a triangular band
// coefficient matrix.
//
// The solution matrix X must be computed by DTBTRS or some other
// means before entering this routine.  Dtbrfs does not do iterative
// refinement because doing so cannot improve the backward error.
func Dtbrfs(uplo mat.MatUplo, trans mat.MatTrans, diag mat.MatDiag, n, kd, nrhs int, ab, b, x *mat.Matrix, ferr, berr, work *mat.Vector, iwork *[]int) (err error) {
	var notran, nounit, upper bool
	var transt mat.MatTrans
	var eps, lstres, one, s, safe1, safe2, safmin, xk, zero float64
	var i, j, k, kase, nz int

	isave := make([]int, 3)

	zero = 0.0
	one = 1.0

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
	} else if kd < 0 {
		err = fmt.Errorf("kd < 0: kd=%v", kd)
	} else if nrhs < 0 {
		err = fmt.Errorf("nrhs < 0: nrhs=%v", nrhs)
	} else if ab.Rows < kd+1 {
		err = fmt.Errorf("ab.Rows < kd+1: ab.Rows=%v, kd=%v", ab.Rows, kd)
	} else if b.Rows < max(1, n) {
		err = fmt.Errorf("b.Rows < max(1, n): b.Rows=%v, n=%v", b.Rows, n)
	} else if x.Rows < max(1, n) {
		err = fmt.Errorf("x.Rows < max(1, n): x.Rows=%v, n=%v", x.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Dtbrfs", err)
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
		transt = Trans
	} else {
		transt = NoTrans
	}

	//     NZ = maximum number of nonzero elements in each row of A, plus 1
	nz = kd + 2
	eps = Dlamch(Epsilon)
	safmin = Dlamch(SafeMinimum)
	safe1 = float64(nz) * safmin
	safe2 = safe1 / eps

	//     Do for each right hand side
	for j = 1; j <= nrhs; j++ {
		//        Compute residual R = B - op(A) * X,
		//        where op(A) = A or A**T, depending on TRANS.
		goblas.Dcopy(n, x.Vector(0, j-1, 1), work.Off(n, 1))
		if err = goblas.Dtbmv(uplo, trans, diag, n, kd, ab, work.Off(n, 1)); err != nil {
			panic(err)
		}
		goblas.Daxpy(n, -one, b.Vector(0, j-1, 1), work.Off(n, 1))

		//        Compute componentwise relative backward error from formula
		//
		//        max(i) ( abs(R(i)) / ( abs(op(A))*abs(X) + abs(B) )(i) )
		//
		//        where abs(Z) is the componentwise absolute value of the matrix
		//        or vector Z.  If the i-th component of the denominator is less
		//        than SAFE2, then SAFE1 is added to the i-th components of the
		//        numerator and denominator before dividing.
		for i = 1; i <= n; i++ {
			work.Set(i-1, math.Abs(b.Get(i-1, j-1)))
		}

		if notran {
			//           Compute abs(A)*abs(X) + abs(B).
			if upper {
				if nounit {
					for k = 1; k <= n; k++ {
						xk = math.Abs(x.Get(k-1, j-1))
						for i = max(1, k-kd); i <= k; i++ {
							work.Set(i-1, work.Get(i-1)+math.Abs(ab.Get(kd+1+i-k-1, k-1))*xk)
						}
					}
				} else {
					for k = 1; k <= n; k++ {
						xk = math.Abs(x.Get(k-1, j-1))
						for i = max(1, k-kd); i <= k-1; i++ {
							work.Set(i-1, work.Get(i-1)+math.Abs(ab.Get(kd+1+i-k-1, k-1))*xk)
						}
						work.Set(k-1, work.Get(k-1)+xk)
					}
				}
			} else {
				if nounit {
					for k = 1; k <= n; k++ {
						xk = math.Abs(x.Get(k-1, j-1))
						for i = k; i <= min(n, k+kd); i++ {
							work.Set(i-1, work.Get(i-1)+math.Abs(ab.Get(1+i-k-1, k-1))*xk)
						}
					}
				} else {
					for k = 1; k <= n; k++ {
						xk = math.Abs(x.Get(k-1, j-1))
						for i = k + 1; i <= min(n, k+kd); i++ {
							work.Set(i-1, work.Get(i-1)+math.Abs(ab.Get(1+i-k-1, k-1))*xk)
						}
						work.Set(k-1, work.Get(k-1)+xk)
					}
				}
			}
		} else {
			//           Compute abs(A**T)*abs(X) + abs(B).
			if upper {
				if nounit {
					for k = 1; k <= n; k++ {
						s = zero
						for i = max(1, k-kd); i <= k; i++ {
							s = s + math.Abs(ab.Get(kd+1+i-k-1, k-1))*math.Abs(x.Get(i-1, j-1))
						}
						work.Set(k-1, work.Get(k-1)+s)
					}
				} else {
					for k = 1; k <= n; k++ {
						s = math.Abs(x.Get(k-1, j-1))
						for i = max(1, k-kd); i <= k-1; i++ {
							s = s + math.Abs(ab.Get(kd+1+i-k-1, k-1))*math.Abs(x.Get(i-1, j-1))
						}
						work.Set(k-1, work.Get(k-1)+s)
					}
				}
			} else {
				if nounit {
					for k = 1; k <= n; k++ {
						s = zero
						for i = k; i <= min(n, k+kd); i++ {
							s = s + math.Abs(ab.Get(1+i-k-1, k-1))*math.Abs(x.Get(i-1, j-1))
						}
						work.Set(k-1, work.Get(k-1)+s)
					}
				} else {
					for k = 1; k <= n; k++ {
						s = math.Abs(x.Get(k-1, j-1))
						for i = k + 1; i <= min(n, k+kd); i++ {
							s = s + math.Abs(ab.Get(1+i-k-1, k-1))*math.Abs(x.Get(i-1, j-1))
						}
						work.Set(k-1, work.Get(k-1)+s)
					}
				}
			}
		}
		s = zero
		for i = 1; i <= n; i++ {
			if work.Get(i-1) > safe2 {
				s = math.Max(s, math.Abs(work.Get(n+i-1))/work.Get(i-1))
			} else {
				s = math.Max(s, (math.Abs(work.Get(n+i-1))+safe1)/(work.Get(i-1)+safe1))
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
		//        Use DLACN2 to estimate the infinity-norm of the matrix
		//           inv(op(A)) * diag(W),
		//        where W = abs(R) + NZ*EPS*( abs(op(A))*abs(X)+abs(B) )))
		for i = 1; i <= n; i++ {
			if work.Get(i-1) > safe2 {
				work.Set(i-1, math.Abs(work.Get(n+i-1))+float64(nz)*eps*work.Get(i-1))
			} else {
				work.Set(i-1, math.Abs(work.Get(n+i-1))+float64(nz)*eps*work.Get(i-1)+safe1)
			}
		}

		kase = 0
	label210:
		;
		_ferr := ferr.GetPtr(j - 1)
		*_ferr, kase = Dlacn2(n, work.Off(2*n), work.Off(n), iwork, ferr.Get(j-1), kase, &isave)
		if kase != 0 {
			if kase == 1 {
				//              Multiply by diag(W)*inv(op(A)**T).
				if err = goblas.Dtbsv(uplo, transt, diag, n, kd, ab, work.Off(n, 1)); err != nil {
					panic(err)
				}
				for i = 1; i <= n; i++ {
					work.Set(n+i-1, work.Get(i-1)*work.Get(n+i-1))
				}
			} else {
				//              Multiply by inv(op(A))*diag(W).
				for i = 1; i <= n; i++ {
					work.Set(n+i-1, work.Get(i-1)*work.Get(n+i-1))
				}
				if err = goblas.Dtbsv(uplo, trans, diag, n, kd, ab, work.Off(n, 1)); err != nil {
					panic(err)
				}
			}
			goto label210
		}

		//        Normalize error.
		lstres = zero
		for i = 1; i <= n; i++ {
			lstres = math.Max(lstres, math.Abs(x.Get(i-1, j-1)))
		}
		if lstres != zero {
			ferr.Set(j-1, ferr.Get(j-1)/lstres)
		}

	}

	return
}
