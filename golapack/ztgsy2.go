package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Ztgsy2 solves the generalized Sylvester equation
//
//             A * R - L * B = scale * C               (1)
//             D * R - L * E = scale * F
//
// using Level 1 and 2 BLAS, where R and L are unknown M-by-N matrices,
// (A, D), (B, E) and (C, F) are given matrix pairs of size M-by-M,
// N-by-N and M-by-N, respectively. A, B, D and E are upper triangular
// (i.e., (A,D) and (B,E) in generalized Schur form).
//
// The solution (R, L) overwrites (C, F). 0 <= SCALE <= 1 is an output
// scaling factor chosen to avoid overflow.
//
// In matrix notation solving equation (1) corresponds to solve
// Zx = scale * b, where Z is defined as
//
//        Z = [ kron(In, A)  -kron(B**H, Im) ]             (2)
//            [ kron(In, D)  -kron(E**H, Im) ],
//
// Ik is the identity matrix of size k and X**H is the conjuguate transpose of X.
// kron(X, Y) is the Kronecker product between the matrices X and Y.
//
// If TRANS = 'C', y in the conjugate transposed system Z**H*y = scale*b
// is solved for, which is equivalent to solve for R and L in
//
//             A**H * R  + D**H * L   = scale * C           (3)
//             R  * B**H + L  * E**H  = scale * -F
//
// This case is used to compute an estimate of Dif[(A, D), (B, E)] =
// = sigma_min(Z) using reverse communication with ZLACON.
//
// ZTGSY2 also (IJOB >= 1) contributes to the computation in ZTGSYL
// of an upper bound on the separation between to matrix pairs. Then
// the input (A, D), (B, E) are sub-pencils of two matrix pairs in
// ZTGSYL.
func Ztgsy2(trans byte, ijob, m, n *int, a *mat.CMatrix, lda *int, b *mat.CMatrix, ldb *int, c *mat.CMatrix, ldc *int, d *mat.CMatrix, ldd *int, e *mat.CMatrix, lde *int, f *mat.CMatrix, ldf *int, scale, rdsum, rdscal *float64, info *int) {
	var notran bool
	var alpha complex128
	var one, scaloc, zero float64
	var i, ierr, j, k, ldz int
	rhs := cvf(2)
	ipiv := make([]int, 2)
	jpiv := make([]int, 2)
	z := cmf(2, 2, opts)

	zero = 0.0
	one = 1.0
	ldz = 2

	//     Decode and test input parameters
	(*info) = 0
	ierr = 0
	notran = trans == 'N'
	if !notran && trans != 'C' {
		(*info) = -1
	} else if notran {
		if ((*ijob) < 0) || ((*ijob) > 2) {
			(*info) = -2
		}
	}
	if (*info) == 0 {
		if (*m) <= 0 {
			(*info) = -3
		} else if (*n) <= 0 {
			(*info) = -4
		} else if (*lda) < maxint(1, *m) {
			(*info) = -6
		} else if (*ldb) < maxint(1, *n) {
			(*info) = -8
		} else if (*ldc) < maxint(1, *m) {
			(*info) = -10
		} else if (*ldd) < maxint(1, *m) {
			(*info) = -12
		} else if (*lde) < maxint(1, *n) {
			(*info) = -14
		} else if (*ldf) < maxint(1, *m) {
			(*info) = -16
		}
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZTGSY2"), -(*info))
		return
	}

	if notran {
		//        Solve (I, J) - system
		//           A(I, I) * R(I, J) - L(I, J) * B(J, J) = C(I, J)
		//           D(I, I) * R(I, J) - L(I, J) * E(J, J) = F(I, J)
		//        for I = M, M - 1, ..., 1; J = 1, 2, ..., N
		(*scale) = one
		scaloc = one
		for j = 1; j <= (*n); j++ {
			for i = (*m); i >= 1; i-- {
				//              Build 2 by 2 system
				z.Set(0, 0, a.Get(i-1, i-1))
				z.Set(1, 0, d.Get(i-1, i-1))
				z.Set(0, 1, -b.Get(j-1, j-1))
				z.Set(1, 1, -e.Get(j-1, j-1))

				//              Set up right hand side(s)
				rhs.Set(0, c.Get(i-1, j-1))
				rhs.Set(1, f.Get(i-1, j-1))

				//              Solve Z * x = RHS
				Zgetc2(&ldz, z, &ldz, &ipiv, &jpiv, &ierr)
				if ierr > 0 {
					(*info) = ierr
				}
				if (*ijob) == 0 {
					Zgesc2(&ldz, z, &ldz, rhs, &ipiv, &jpiv, &scaloc)
					if scaloc != one {
						for k = 1; k <= (*n); k++ {
							goblas.Zscal(*m, complex(scaloc, zero), c.CVector(0, k-1), 1)
							goblas.Zscal(*m, complex(scaloc, zero), f.CVector(0, k-1), 1)
						}
						(*scale) = (*scale) * scaloc
					}
				} else {
					Zlatdf(ijob, &ldz, z, &ldz, rhs, rdsum, rdscal, &ipiv, &jpiv)
				}

				//              Unpack solution vector(s)
				c.Set(i-1, j-1, rhs.Get(0))
				f.Set(i-1, j-1, rhs.Get(1))

				//              Substitute R(I, J) and L(I, J) into remaining equation.
				if i > 1 {
					alpha = -rhs.Get(0)
					goblas.Zaxpy(i-1, alpha, a.CVector(0, i-1), 1, c.CVector(0, j-1), 1)
					goblas.Zaxpy(i-1, alpha, d.CVector(0, i-1), 1, f.CVector(0, j-1), 1)
				}
				if j < (*n) {
					goblas.Zaxpy((*n)-j, rhs.Get(1), b.CVector(j-1, j+1-1), *ldb, c.CVector(i-1, j+1-1), *ldc)
					goblas.Zaxpy((*n)-j, rhs.Get(1), e.CVector(j-1, j+1-1), *lde, f.CVector(i-1, j+1-1), *ldf)
				}

			}
		}
	} else {
		//        Solve transposed (I, J) - system:
		//           A(I, I)**H * R(I, J) + D(I, I)**H * L(J, J) = C(I, J)
		//           R(I, I) * B(J, J) + L(I, J) * E(J, J)   = -F(I, J)
		//        for I = 1, 2, ..., M, J = N, N - 1, ..., 1
		(*scale) = one
		scaloc = one
		for i = 1; i <= (*m); i++ {
			for j = (*n); j >= 1; j-- {
				//              Build 2 by 2 system Z**H
				z.Set(0, 0, a.GetConj(i-1, i-1))
				z.Set(1, 0, -b.GetConj(j-1, j-1))
				z.Set(0, 1, d.GetConj(i-1, i-1))
				z.Set(1, 1, -e.GetConj(j-1, j-1))

				//              Set up right hand side(s)
				rhs.Set(0, c.Get(i-1, j-1))
				rhs.Set(1, f.Get(i-1, j-1))

				//              Solve Z**H * x = RHS
				Zgetc2(&ldz, z, &ldz, &ipiv, &jpiv, &ierr)
				if ierr > 0 {
					(*info) = ierr
				}
				Zgesc2(&ldz, z, &ldz, rhs, &ipiv, &jpiv, &scaloc)
				if scaloc != one {
					for k = 1; k <= (*n); k++ {
						goblas.Zscal(*m, complex(scaloc, zero), c.CVector(0, k-1), 1)
						goblas.Zscal(*m, complex(scaloc, zero), f.CVector(0, k-1), 1)
					}
					(*scale) = (*scale) * scaloc
				}

				//              Unpack solution vector(s)
				c.Set(i-1, j-1, rhs.Get(0))
				f.Set(i-1, j-1, rhs.Get(1))

				//              Substitute R(I, J) and L(I, J) into remaining equation.
				for k = 1; k <= j-1; k++ {
					f.Set(i-1, k-1, f.Get(i-1, k-1)+rhs.Get(0)*b.GetConj(k-1, j-1)+rhs.Get(1)*e.GetConj(k-1, j-1))
				}
				for k = i + 1; k <= (*m); k++ {
					c.Set(k-1, j-1, c.Get(k-1, j-1)-a.GetConj(i-1, k-1)*rhs.Get(0)-d.GetConj(i-1, k-1)*rhs.Get(1))
				}

			}
		}
	}
}
