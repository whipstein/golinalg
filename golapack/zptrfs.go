package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zptrfs improves the computed solution to a system of linear
// equations when the coefficient matrix is Hermitian positive definite
// and tridiagonal, and provides error bounds and backward error
// estimates for the solution.
func Zptrfs(uplo mat.MatUplo, n, nrhs int, d *mat.Vector, e *mat.CVector, df *mat.Vector, ef *mat.CVector, b, x *mat.CMatrix, ferr, berr *mat.Vector, work *mat.CVector, rwork *mat.Vector) (err error) {
	var upper bool
	var bi, cx, dx, ex complex128
	var eps, lstres, one, s, safe1, safe2, safmin, three, two, zero float64
	var count, i, itmax, ix, j, nz int

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
		gltest.Xerbla2("Zptrfs", err)
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
		//        Compute residual R = B - A * X.  Also compute
		//        abs(A)*abs(x) + abs(b) for use in the backward error bound.
		if upper {
			if n == 1 {
				bi = b.Get(0, j-1)
				dx = d.GetCmplx(0) * x.Get(0, j-1)
				work.Set(0, bi-dx)
				rwork.Set(0, cabs1(bi)+cabs1(dx))
			} else {
				bi = b.Get(0, j-1)
				dx = d.GetCmplx(0) * x.Get(0, j-1)
				ex = e.Get(0) * x.Get(1, j-1)
				work.Set(0, bi-dx-ex)
				rwork.Set(0, cabs1(bi)+cabs1(dx)+cabs1(e.Get(0))*cabs1(x.Get(1, j-1)))
				for i = 2; i <= n-1; i++ {
					bi = b.Get(i-1, j-1)
					cx = e.GetConj(i-1-1) * x.Get(i-1-1, j-1)
					dx = d.GetCmplx(i-1) * x.Get(i-1, j-1)
					ex = e.Get(i-1) * x.Get(i, j-1)
					work.Set(i-1, bi-cx-dx-ex)
					rwork.Set(i-1, cabs1(bi)+cabs1(e.Get(i-1-1))*cabs1(x.Get(i-1-1, j-1))+cabs1(dx)+cabs1(e.Get(i-1))*cabs1(x.Get(i, j-1)))
				}
				bi = b.Get(n-1, j-1)
				cx = e.GetConj(n-1-1) * x.Get(n-1-1, j-1)
				dx = d.GetCmplx(n-1) * x.Get(n-1, j-1)
				work.Set(n-1, bi-cx-dx)
				rwork.Set(n-1, cabs1(bi)+cabs1(e.Get(n-1-1))*cabs1(x.Get(n-1-1, j-1))+cabs1(dx))
			}
		} else {
			if n == 1 {
				bi = b.Get(0, j-1)
				dx = d.GetCmplx(0) * x.Get(0, j-1)
				work.Set(0, bi-dx)
				rwork.Set(0, cabs1(bi)+cabs1(dx))
			} else {
				bi = b.Get(0, j-1)
				dx = d.GetCmplx(0) * x.Get(0, j-1)
				ex = e.GetConj(0) * x.Get(1, j-1)
				work.Set(0, bi-dx-ex)
				rwork.Set(0, cabs1(bi)+cabs1(dx)+cabs1(e.Get(0))*cabs1(x.Get(1, j-1)))
				for i = 2; i <= n-1; i++ {
					bi = b.Get(i-1, j-1)
					cx = e.Get(i-1-1) * x.Get(i-1-1, j-1)
					dx = d.GetCmplx(i-1) * x.Get(i-1, j-1)
					ex = e.GetConj(i-1) * x.Get(i, j-1)
					work.Set(i-1, bi-cx-dx-ex)
					rwork.Set(i-1, cabs1(bi)+cabs1(e.Get(i-1-1))*cabs1(x.Get(i-1-1, j-1))+cabs1(dx)+cabs1(e.Get(i-1))*cabs1(x.Get(i, j-1)))
				}
				bi = b.Get(n-1, j-1)
				cx = e.Get(n-1-1) * x.Get(n-1-1, j-1)
				dx = d.GetCmplx(n-1) * x.Get(n-1, j-1)
				work.Set(n-1, bi-cx-dx)
				rwork.Set(n-1, cabs1(bi)+cabs1(e.Get(n-1-1))*cabs1(x.Get(n-1-1, j-1))+cabs1(dx))
			}
		}

		//        Compute componentwise relative backward error from formula
		//
		//        max(i) ( abs(R(i)) / ( abs(A)*abs(X) + abs(B) )(i) )
		//
		//        where abs(Z) is the componentwise absolute value of the matrix
		//        or vector Z.  If the i-th component of the denominator is less
		//        than SAFE2, then SAFE1 is added to the i-th components of the
		//        numerator and denominator before dividing.
		s = zero
		for i = 1; i <= n; i++ {
			if rwork.Get(i-1) > safe2 {
				s = math.Max(s, cabs1(work.Get(i-1))/rwork.Get(i-1))
			} else {
				s = math.Max(s, (cabs1(work.Get(i-1))+safe1)/(rwork.Get(i-1)+safe1))
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
			if err = Zpttrs(uplo, n, 1, df, ef, work.CMatrix(n, opts)); err != nil {
				panic(err)
			}
			x.Off(0, j-1).CVector().Axpy(n, complex(one, 0), work, 1, 1)
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
		for i = 1; i <= n; i++ {
			if rwork.Get(i-1) > safe2 {
				rwork.Set(i-1, cabs1(work.Get(i-1))+float64(nz)*eps*rwork.Get(i-1))
			} else {
				rwork.Set(i-1, cabs1(work.Get(i-1))+float64(nz)*eps*rwork.Get(i-1)+safe1)
			}
		}
		ix = rwork.Iamax(n, 1)
		ferr.Set(j-1, rwork.Get(ix-1))

		//        Estimate the norm of inv(A).
		//
		//        Solve M(A) * x = e, where M(A) = (m(i,j)) is given by
		//
		//           m(i,j) =  abs(A(i,j)), i = j,
		//           m(i,j) = -abs(A(i,j)), i .ne. j,
		//
		//        and e = [ 1, 1, ..., 1 ]**T.  Note M(A) = M(L)*D*M(L)**H.
		//
		//        Solve M(L) * x = e.
		rwork.Set(0, one)
		for i = 2; i <= n; i++ {
			rwork.Set(i-1, one+rwork.Get(i-1-1)*ef.GetMag(i-1-1))
		}

		//        Solve D * M(L)**H * x = b.
		rwork.Set(n-1, rwork.Get(n-1)/df.Get(n-1))
		for i = n - 1; i >= 1; i-- {
			rwork.Set(i-1, rwork.Get(i-1)/df.Get(i-1)+rwork.Get(i)*ef.GetMag(i-1))
		}

		//        Compute norm(inv(A)) = max(x(i)), 1<=i<=n.
		ix = rwork.Iamax(n, 1)
		ferr.Set(j-1, ferr.Get(j-1)*rwork.GetMag(ix-1))

		//        Normalize error.
		lstres = zero
		for i = 1; i <= n; i++ {
			lstres = math.Max(lstres, x.GetMag(i-1, j-1))
		}
		if lstres != zero {
			ferr.Set(j-1, ferr.Get(j-1)/lstres)
		}

	}

	return
}
