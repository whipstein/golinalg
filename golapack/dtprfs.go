package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dtprfs provides error bounds and backward error estimates for the
// solution to a system of linear equations with a triangular packed
// coefficient matrix.
//
// The solution matrix X must be computed by DTPTRS or some other
// means before entering this routine.  DTPRFS does not do iterative
// refinement because doing so cannot improve the backward error.
func Dtprfs(uplo, trans, diag byte, n, nrhs *int, ap *mat.Vector, b *mat.Matrix, ldb *int, x *mat.Matrix, ldx *int, ferr, berr, work *mat.Vector, iwork *[]int, info *int) {
	var notran, nounit, upper bool
	var transt byte
	var eps, lstres, one, s, safe1, safe2, safmin, xk, zero float64
	var i, j, k, kase, kc, nz int
	isave := make([]int, 3)
	var err error
	_ = err

	zero = 0.0
	one = 1.0

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
		gltest.Xerbla([]byte("DTPRFS"), -(*info))
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
		transt = 'T'
	} else {
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
		//        where op(A) = A or A**T, depending on TRANS.
		goblas.Dcopy(*n, x.Vector(0, j-1), 1, work.Off((*n)+1-1), 1)
		err = goblas.Dtpmv(mat.UploByte(uplo), mat.TransByte(trans), mat.DiagByte(diag), *n, ap, work.Off((*n)+1-1), 1)
		goblas.Daxpy(*n, -one, b.Vector(0, j-1), 1, work.Off((*n)+1-1), 1)

		//        Compute componentwise relative backward error from formula
		//
		//        max(i) ( abs(R(i)) / ( abs(op(A))*abs(X) + abs(B) )(i) )
		//
		//        where abs(Z) is the componentwise absolute value of the matrix
		//        or vector Z.  If the i-th component of the denominator is less
		//        than SAFE2, then SAFE1 is added to the i-th components of the
		//        numerator and denominator before dividing.
		for i = 1; i <= (*n); i++ {
			work.Set(i-1, math.Abs(b.Get(i-1, j-1)))
		}

		if notran {
			//           Compute abs(A)*abs(X) + abs(B).
			if upper {
				kc = 1
				if nounit {
					for k = 1; k <= (*n); k++ {
						xk = math.Abs(x.Get(k-1, j-1))
						for i = 1; i <= k; i++ {
							work.Set(i-1, work.Get(i-1)+math.Abs(ap.Get(kc+i-1-1))*xk)
						}
						kc = kc + k
					}
				} else {
					for k = 1; k <= (*n); k++ {
						xk = math.Abs(x.Get(k-1, j-1))
						for i = 1; i <= k-1; i++ {
							work.Set(i-1, work.Get(i-1)+math.Abs(ap.Get(kc+i-1-1))*xk)
						}
						work.Set(k-1, work.Get(k-1)+xk)
						kc = kc + k
					}
				}
			} else {
				kc = 1
				if nounit {
					for k = 1; k <= (*n); k++ {
						xk = math.Abs(x.Get(k-1, j-1))
						for i = k; i <= (*n); i++ {
							work.Set(i-1, work.Get(i-1)+math.Abs(ap.Get(kc+i-k-1))*xk)
						}
						kc = kc + (*n) - k + 1
					}
				} else {
					for k = 1; k <= (*n); k++ {
						xk = math.Abs(x.Get(k-1, j-1))
						for i = k + 1; i <= (*n); i++ {
							work.Set(i-1, work.Get(i-1)+math.Abs(ap.Get(kc+i-k-1))*xk)
						}
						work.Set(k-1, work.Get(k-1)+xk)
						kc = kc + (*n) - k + 1
					}
				}
			}
		} else {
			//           Compute abs(A**T)*abs(X) + abs(B).
			if upper {
				kc = 1
				if nounit {
					for k = 1; k <= (*n); k++ {
						s = zero
						for i = 1; i <= k; i++ {
							s = s + math.Abs(ap.Get(kc+i-1-1))*math.Abs(x.Get(i-1, j-1))
						}
						work.Set(k-1, work.Get(k-1)+s)
						kc = kc + k
					}
				} else {
					for k = 1; k <= (*n); k++ {
						s = math.Abs(x.Get(k-1, j-1))
						for i = 1; i <= k-1; i++ {
							s = s + math.Abs(ap.Get(kc+i-1-1))*math.Abs(x.Get(i-1, j-1))
						}
						work.Set(k-1, work.Get(k-1)+s)
						kc = kc + k
					}
				}
			} else {
				kc = 1
				if nounit {
					for k = 1; k <= (*n); k++ {
						s = zero
						for i = k; i <= (*n); i++ {
							s = s + math.Abs(ap.Get(kc+i-k-1))*math.Abs(x.Get(i-1, j-1))
						}
						work.Set(k-1, work.Get(k-1)+s)
						kc = kc + (*n) - k + 1
					}
				} else {
					for k = 1; k <= (*n); k++ {
						s = math.Abs(x.Get(k-1, j-1))
						for i = k + 1; i <= (*n); i++ {
							s = s + math.Abs(ap.Get(kc+i-k-1))*math.Abs(x.Get(i-1, j-1))
						}
						work.Set(k-1, work.Get(k-1)+s)
						kc = kc + (*n) - k + 1
					}
				}
			}
		}
		s = zero
		for i = 1; i <= (*n); i++ {
			if work.Get(i-1) > safe2 {
				s = maxf64(s, math.Abs(work.Get((*n)+i-1))/work.Get(i-1))
			} else {
				s = maxf64(s, (math.Abs(work.Get((*n)+i-1))+safe1)/(work.Get(i-1)+safe1))
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
		for i = 1; i <= (*n); i++ {
			if work.Get(i-1) > safe2 {
				work.Set(i-1, math.Abs(work.Get((*n)+i-1))+float64(nz)*eps*work.Get(i-1))
			} else {
				work.Set(i-1, math.Abs(work.Get((*n)+i-1))+float64(nz)*eps*work.Get(i-1)+safe1)
			}
		}

		kase = 0
	label210:
		;
		Dlacn2(n, work.Off(2*(*n)+1-1), work.Off((*n)+1-1), iwork, ferr.GetPtr(j-1), &kase, &isave)
		if kase != 0 {
			if kase == 1 {
				//              Multiply by diag(W)*inv(op(A)**T).
				err = goblas.Dtpsv(mat.UploByte(uplo), mat.TransByte(transt), mat.DiagByte(diag), *n, ap, work.Off((*n)+1-1), 1)
				for i = 1; i <= (*n); i++ {
					work.Set((*n)+i-1, work.Get(i-1)*work.Get((*n)+i-1))
				}
			} else {
				//              Multiply by inv(op(A))*diag(W).
				for i = 1; i <= (*n); i++ {
					work.Set((*n)+i-1, work.Get(i-1)*work.Get((*n)+i-1))
				}
				err = goblas.Dtpsv(mat.UploByte(uplo), mat.TransByte(trans), mat.DiagByte(diag), *n, ap, work.Off((*n)+1-1), 1)
			}
			goto label210
		}

		//        Normalize error.
		lstres = zero
		for i = 1; i <= (*n); i++ {
			lstres = maxf64(lstres, math.Abs(x.Get(i-1, j-1)))
		}
		if lstres != zero {
			ferr.Set(j-1, ferr.Get(j-1)/lstres)
		}

	}
}
