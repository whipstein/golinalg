package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dpprfs improves the computed solution to a system of linear
// equations when the coefficient matrix is symmetric positive definite
// and packed, and provides error bounds and backward error estimates
// for the solution.
func Dpprfs(uplo mat.MatUplo, n, nrhs int, ap, afp *mat.Vector, b, x *mat.Matrix, ferr, berr, work *mat.Vector, iwork *[]int) (err error) {
	var upper bool
	var eps, lstres, one, s, safe1, safe2, safmin, three, two, xk, zero float64
	var count, i, ik, itmax, j, k, kase, kk, nz int

	isave := make([]int, 3)

	itmax = 5
	zero = 0.0
	one = 1.0
	two = 2.0
	three = 3.0

	//     Test the input parameters.
	upper = uplo == Upper
	if !upper && uplo != Lower {
		err = fmt.Errorf("!upper && uplo != Lower: uplo=%s", uplo)
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
		gltest.Xerbla2("Dpprfs", err)
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

	//     NZ = maximum number of nonzero elements in each row of A, plus 1
	nz = n + 1
	eps = Dlamch(Epsilon)
	safmin = Dlamch(SafeMinimum)
	safe1 = float64(nz) * safmin
	safe2 = safe1 / eps

	//     Do for each right hand side
	for j = 1; j <= nrhs; j++ {

		count = 1
		lstres = three
	label20:
		;

		//        Loop until stopping criterion is satisfied.
		//
		//        Compute residual R = B - A * X
		goblas.Dcopy(n, b.Vector(0, j-1, 1), work.Off(n, 1))
		if err = goblas.Dspmv(uplo, n, -one, ap, x.Vector(0, j-1, 1), one, work.Off(n, 1)); err != nil {
			panic(err)
		}

		//        Compute componentwise relative backward error from formula
		//
		//        max(i) ( abs(R(i)) / ( abs(A)*abs(X) + abs(B) )(i) )
		//
		//        where abs(Z) is the componentwise absolute value of the matrix
		//        or vector Z.  If the i-th component of the denominator is less
		//        than SAFE2, then SAFE1 is added to the i-th components of the
		//        numerator and denominator before dividing.
		for i = 1; i <= n; i++ {
			work.Set(i-1, math.Abs(b.Get(i-1, j-1)))
		}

		//        Compute abs(A)*abs(X) + abs(B).
		kk = 1
		if upper {
			for k = 1; k <= n; k++ {
				s = zero
				xk = math.Abs(x.Get(k-1, j-1))
				ik = kk
				for i = 1; i <= k-1; i++ {
					work.Set(i-1, work.Get(i-1)+math.Abs(ap.Get(ik-1))*xk)
					s = s + math.Abs(ap.Get(ik-1))*math.Abs(x.Get(i-1, j-1))
					ik = ik + 1
				}
				work.Set(k-1, work.Get(k-1)+math.Abs(ap.Get(kk+k-1-1))*xk+s)
				kk = kk + k
			}
		} else {
			for k = 1; k <= n; k++ {
				s = zero
				xk = math.Abs(x.Get(k-1, j-1))
				work.Set(k-1, work.Get(k-1)+math.Abs(ap.Get(kk-1))*xk)
				ik = kk + 1
				for i = k + 1; i <= n; i++ {
					work.Set(i-1, work.Get(i-1)+math.Abs(ap.Get(ik-1))*xk)
					s = s + math.Abs(ap.Get(ik-1))*math.Abs(x.Get(i-1, j-1))
					ik = ik + 1
				}
				work.Set(k-1, work.Get(k-1)+s)
				kk = kk + (n - k + 1)
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

		//        Test stopping criterion. Continue iterating if
		//           1) The residual BERR(J) is larger than machine epsilon, and
		//           2) BERR(J) decreased by at least a factor of 2 during the
		//              last iteration, and
		//           3) At most ITMAX iterations tried.
		if berr.Get(j-1) > eps && two*berr.Get(j-1) <= lstres && count <= itmax {
			//           Update solution and try again.
			if err = Dpptrs(uplo, n, 1, afp, work.MatrixOff(n, n, opts)); err != nil {
				panic(err)
			}
			goblas.Daxpy(n, one, work.Off(n, 1), x.Vector(0, j-1, 1))
			lstres = berr.Get(j - 1)
			count = count + 1
			goto label20
		}

		//        Bound error from formula
		//
		//        norm(X - XTRUE) / norm(X) .le. FERR =
		//        norm( abs(inv(A))*
		//           ( abs(R) + NZ*EPS*( abs(A)*abs(X)+abs(B) ))) / norm(X)
		//
		//        where
		//          norm(Z) is the magnitude of the largest component of Z
		//          inv(A) is the inverse of A
		//          abs(Z) is the componentwise absolute value of the matrix or
		//             vector Z
		//          NZ is the maximum number of nonzeros in any row of A, plus 1
		//          EPS is machine epsilon
		//
		//        The i-th component of abs(R)+NZ*EPS*(abs(A)*abs(X)+abs(B))
		//        is incremented by SAFE1 if the i-th component of
		//        abs(A)*abs(X) + abs(B) is less than SAFE2.
		//
		//        Use DLACN2 to estimate the infinity-norm of the matrix
		//           inv(A) * diag(W),
		//        where W = abs(R) + NZ*EPS*( abs(A)*abs(X)+abs(B) )))
		for i = 1; i <= n; i++ {
			if work.Get(i-1) > safe2 {
				work.Set(i-1, math.Abs(work.Get(n+i-1))+float64(nz)*eps*work.Get(i-1))
			} else {
				work.Set(i-1, math.Abs(work.Get(n+i-1))+float64(nz)*eps*work.Get(i-1)+safe1)
			}
		}

		kase = 0
	label100:
		;
		_ferr := ferr.GetPtr(j - 1)
		*_ferr, kase = Dlacn2(n, work.Off(2*n), work.Off(n), iwork, ferr.Get(j-1), kase, &isave)
		if kase != 0 {
			if kase == 1 {
				//              Multiply by diag(W)*inv(A**T).
				if err = Dpptrs(uplo, n, 1, afp, work.MatrixOff(n, n, opts)); err != nil {
					panic(err)
				}
				for i = 1; i <= n; i++ {
					work.Set(n+i-1, work.Get(i-1)*work.Get(n+i-1))
				}
			} else if kase == 2 {
				//
				//              Multiply by inv(A)*diag(W).
				//
				for i = 1; i <= n; i++ {
					work.Set(n+i-1, work.Get(i-1)*work.Get(n+i-1))
				}
				if err = Dpptrs(uplo, n, 1, afp, work.MatrixOff(n, n, opts)); err != nil {
					panic(err)
				}
			}
			goto label100
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
