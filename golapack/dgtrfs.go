package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgtrfs improves the computed solution to a system of linear
// equations when the coefficient matrix is tridiagonal, and provides
// error bounds and backward error estimates for the solution.
func Dgtrfs(trans mat.MatTrans, n, nrhs int, dl, d, du, dlf, df, duf, du2 *mat.Vector, ipiv []int, b *mat.Matrix, x *mat.Matrix, ferr, berr, work *mat.Vector, iwork *[]int) (err error) {
	var notran bool
	var transn, transt mat.MatTrans
	var eps, lstres, one, s, safe1, safe2, safmin, three, two, zero float64
	var count, i, itmax, j, kase, nz int
	isave := make([]int, 3)

	itmax = 5
	zero = 0.0
	one = 1.0
	two = 2.0
	three = 3.0

	//     Test the input parameters.
	notran = trans == NoTrans
	if !trans.IsValid() {
		err = fmt.Errorf("!trans.IsValid(): trans=%s", trans)
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
		gltest.Xerbla2("Dgtrfs", err)
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
		transt = Trans
	} else {
		transn = Trans
		transt = NoTrans
	}

	//     NZ = maximum number of nonzero elements in each row of A, plus 1
	nz = 4
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
		//        Compute residual R = B - op(A) * X,
		//        where op(A) = A, A**T, or A**H, depending on TRANS.
		work.Off(n).Copy(n, b.Off(0, j-1).Vector(), 1, 1)
		Dlagtm(trans, n, 1, -one, dl, d, du, x.Off(0, j-1), one, work.Off(n).Matrix(n, opts))
		//
		//        Compute math.Abs(op(A))*math.Abs(x) + math.Abs(b) for use in the backward
		//        error bound.
		//
		if notran {
			if n == 1 {
				work.Set(0, math.Abs(b.Get(0, j-1))+math.Abs(d.Get(0)*x.Get(0, j-1)))
			} else {
				work.Set(0, math.Abs(b.Get(0, j-1))+math.Abs(d.Get(0)*x.Get(0, j-1))+math.Abs(du.Get(0)*x.Get(1, j-1)))
				for i = 2; i <= n-1; i++ {
					work.Set(i-1, math.Abs(b.Get(i-1, j-1))+math.Abs(dl.Get(i-1-1)*x.Get(i-1-1, j-1))+math.Abs(d.Get(i-1)*x.Get(i-1, j-1))+math.Abs(du.Get(i-1)*x.Get(i, j-1)))
				}
				work.Set(n-1, math.Abs(b.Get(n-1, j-1))+math.Abs(dl.Get(n-1-1)*x.Get(n-1-1, j-1))+math.Abs(d.Get(n-1)*x.Get(n-1, j-1)))
			}
		} else {
			if n == 1 {
				work.Set(0, math.Abs(b.Get(0, j-1))+math.Abs(d.Get(0)*x.Get(0, j-1)))
			} else {
				work.Set(0, math.Abs(b.Get(0, j-1))+math.Abs(d.Get(0)*x.Get(0, j-1))+math.Abs(dl.Get(0)*x.Get(1, j-1)))
				for i = 2; i <= n-1; i++ {
					work.Set(i-1, math.Abs(b.Get(i-1, j-1))+math.Abs(du.Get(i-1-1)*x.Get(i-1-1, j-1))+math.Abs(d.Get(i-1)*x.Get(i-1, j-1))+math.Abs(dl.Get(i-1)*x.Get(i, j-1)))
				}
				work.Set(n-1, math.Abs(b.Get(n-1, j-1))+math.Abs(du.Get(n-1-1)*x.Get(n-1-1, j-1))+math.Abs(d.Get(n-1)*x.Get(n-1, j-1)))
			}
		}

		//        Compute componentwise relative backward error from formula
		//
		//        max(i) ( math.Abs(R(i)) / ( math.Abs(op(A))*math.Abs(X) + math.Abs(B) )(i) )
		//
		//        where math.Abs(Z) is the componentwise absolute value of the matrix
		//        or vector Z.  If the i-th component of the denominator is less
		//        than SAFE2, then SAFE1 is added to the i-th components of the
		//        numerator and denominator before dividing.
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
			if err = Dgttrs(trans, n, 1, dlf, df, duf, du2, ipiv, work.Off(n).Matrix(n, opts)); err != nil {
				panic(err)
			}
			x.Off(0, j-1).Vector().Axpy(n, one, work.Off(n), 1, 1)
			lstres = berr.Get(j - 1)
			count = count + 1
			goto label20
		}

		//        Bound error from formula
		//
		//        norm(X - XTRUE) / norm(X) .le. FERR =
		//        norm( math.Abs(inv(op(A)))*
		//           ( math.Abs(R) + NZ*EPS*( math.Abs(op(A))*math.Abs(X)+math.Abs(B) ))) / norm(X)
		//
		//        where
		//          norm(Z) is the magnitude of the largest component of Z
		//          inv(op(A)) is the inverse of op(A)
		//          math.Abs(Z) is the componentwise absolute value of the matrix or
		//             vector Z
		//          NZ is the maximum number of nonzeros in any row of A, plus 1
		//          EPS is machine epsilon
		//
		//        The i-th component of math.Abs(R)+NZ*EPS*(math.Abs(op(A))*math.Abs(X)+math.Abs(B))
		//        is incremented by SAFE1 if the i-th component of
		//        math.Abs(op(A))*math.Abs(X) + math.Abs(B) is less than SAFE2.
		//
		//        Use DLACN2 to estimate the infinity-norm of the matrix
		//           inv(op(A)) * diag(W),
		//        where W = math.Abs(R) + NZ*EPS*( math.Abs(op(A))*math.Abs(X)+math.Abs(B) )))
		for i = 1; i <= n; i++ {
			if work.Get(i-1) > safe2 {
				work.Set(i-1, math.Abs(work.Get(n+i-1))+float64(nz)*eps*work.Get(i-1))
			} else {
				work.Set(i-1, math.Abs(work.Get(n+i-1))+float64(nz)*eps*work.Get(i-1)+safe1)
			}
		}

		kase = 0
	label70:
		;
		_ferr := ferr.GetPtr(j - 1)
		*_ferr, kase = Dlacn2(n, work.Off(2*n), work.Off(n), iwork, ferr.Get(j-1), kase, &isave)
		if kase != 0 {
			if kase == 1 {
				//              Multiply by diag(W)*inv(op(A)**T).
				if err = Dgttrs(transt, n, 1, dlf, df, duf, du2, ipiv, work.Off(n).Matrix(n, opts)); err != nil {
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
				if err = Dgttrs(transn, n, 1, dlf, df, duf, du2, ipiv, work.Off(n).Matrix(n, opts)); err != nil {
					panic(err)
				}
			}
			goto label70
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
