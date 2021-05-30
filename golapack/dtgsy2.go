package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Dtgsy2 solves the generalized Sylvester equation:
//
//             A * R - L * B = scale * C                (1)
//             D * R - L * E = scale * F,
//
// using Level 1 and 2 BLAS. where R and L are unknown M-by-N matrices,
// (A, D), (B, E) and (C, F) are given matrix pairs of size M-by-M,
// N-by-N and M-by-N, respectively, with real entries. (A, D) and (B, E)
// must be in generalized Schur canonical form, i.e. A, B are upper
// quasi triangular and D, E are upper triangular. The solution (R, L)
// overwrites (C, F). 0 <= SCALE <= 1 is an output scaling factor
// chosen to avoid overflow.
//
// In matrix notation solving equation (1) corresponds to solve
// Z*x = scale*b, where Z is defined as
//
//        Z = [ kron(In, A)  -kron(B**T, Im) ]             (2)
//            [ kron(In, D)  -kron(E**T, Im) ],
//
// Ik is the identity matrix of size k and X**T is the transpose of X.
// kron(X, Y) is the Kronecker product between the matrices X and Y.
// In the process of solving (1), we solve a number of such systems
// where Dim(In), Dim(In) = 1 or 2.
//
// If TRANS = 'T', solve the transposed system Z**T*y = scale*b for y,
// which is equivalent to solve for R and L in
//
//             A**T * R  + D**T * L   = scale * C           (3)
//             R  * B**T + L  * E**T  = scale * -F
//
// This case is used to compute an estimate of Dif[(A, D), (B, E)] =
// sigma_min(Z) using reverse communication with DLACON.
//
// DTGSY2 also (IJOB >= 1) contributes to the computation in DTGSYL
// of an upper bound on the separation between to matrix pairs. Then
// the input (A, D), (B, E) are sub-pencils of the matrix pair in
// DTGSYL. See DTGSYL for details.
func Dtgsy2(trans byte, ijob, m, n *int, a *mat.Matrix, lda *int, b *mat.Matrix, ldb *int, c *mat.Matrix, ldc *int, d *mat.Matrix, ldd *int, e *mat.Matrix, lde *int, f *mat.Matrix, ldf *int, scale, rdsum, rdscal *float64, iwork *[]int, pq, info *int) {
	var notran bool
	var alpha, one, scaloc, zero float64
	var i, ie, ierr, ii, is, isp1, j, je, jj, js, jsp1, k, ldz, mb, nb, p, q, zdim int

	rhs := vf(8)
	ipiv := make([]int, 8)
	jpiv := make([]int, 8)
	z := mf(8, 8, opts)

	ldz = 8
	zero = 0.0
	one = 1.0

	//     Decode and test input parameters
	(*info) = 0
	ierr = 0
	notran = trans == 'N'
	if !notran && trans != 'T' {
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
		gltest.Xerbla([]byte("DTGSY2"), -(*info))
		return
	}

	//     Determine block structure of A
	(*pq) = 0
	p = 0
	i = 1
label10:
	;
	if i > (*m) {
		goto label20
	}
	p = p + 1
	(*iwork)[p-1] = i
	if i == (*m) {
		goto label20
	}
	if a.Get(i+1-1, i-1) != zero {
		i = i + 2
	} else {
		i = i + 1
	}
	goto label10
label20:
	;
	(*iwork)[p+1-1] = (*m) + 1

	//     Determine block structure of B
	q = p + 1
	j = 1
label30:
	;
	if j > (*n) {
		goto label40
	}
	q = q + 1
	(*iwork)[q-1] = j
	if j == (*n) {
		goto label40
	}
	if b.Get(j+1-1, j-1) != zero {
		j = j + 2
	} else {
		j = j + 1
	}
	goto label30
label40:
	;
	(*iwork)[q+1-1] = (*n) + 1
	(*pq) = p * (q - p - 1)

	if notran {
		//        Solve (I, J) - subsystem
		//           A(I, I) * R(I, J) - L(I, J) * B(J, J) = C(I, J)
		//           D(I, I) * R(I, J) - L(I, J) * E(J, J) = F(I, J)
		//        for I = P, P - 1, ..., 1; J = 1, 2, ..., Q
		(*scale) = one
		scaloc = one
		for j = p + 2; j <= q; j++ {
			js = (*iwork)[j-1]
			jsp1 = js + 1
			je = (*iwork)[j+1-1] - 1
			nb = je - js + 1
			for i = p; i >= 1; i-- {

				is = (*iwork)[i-1]
				isp1 = is + 1
				ie = (*iwork)[i+1-1] - 1
				mb = ie - is + 1
				zdim = mb * nb * 2

				if (mb == 1) && (nb == 1) {
					//                 Build a 2-by-2 system Z * x = RHS
					z.Set(0, 0, a.Get(is-1, is-1))
					z.Set(1, 0, d.Get(is-1, is-1))
					z.Set(0, 1, -b.Get(js-1, js-1))
					z.Set(1, 1, -e.Get(js-1, js-1))

					//                 Set up right hand side(s)
					rhs.Set(0, c.Get(is-1, js-1))
					rhs.Set(1, f.Get(is-1, js-1))

					//                 Solve Z * x = RHS
					Dgetc2(&zdim, z, &ldz, &ipiv, &jpiv, &ierr)
					if ierr > 0 {
						(*info) = ierr
					}

					if (*ijob) == 0 {
						Dgesc2(&zdim, z, &ldz, rhs, &ipiv, &jpiv, &scaloc)
						if scaloc != one {
							for k = 1; k <= (*n); k++ {
								goblas.Dscal(m, &scaloc, c.Vector(0, k-1), func() *int { y := 1; return &y }())
								goblas.Dscal(m, &scaloc, f.Vector(0, k-1), func() *int { y := 1; return &y }())
							}
							(*scale) = (*scale) * scaloc
						}
					} else {
						Dlatdf(ijob, &zdim, z, &ldz, rhs, rdsum, rdscal, &ipiv, &jpiv)
					}

					//                 Unpack solution vector(s)
					c.Set(is-1, js-1, rhs.Get(0))
					f.Set(is-1, js-1, rhs.Get(1))

					//                 Substitute R(I, J) and L(I, J) into remaining
					//                 equation.
					if i > 1 {
						alpha = -rhs.Get(0)
						goblas.Daxpy(toPtr(is-1), &alpha, a.Vector(0, is-1), func() *int { y := 1; return &y }(), c.Vector(0, js-1), func() *int { y := 1; return &y }())
						goblas.Daxpy(toPtr(is-1), &alpha, d.Vector(0, is-1), func() *int { y := 1; return &y }(), f.Vector(0, js-1), func() *int { y := 1; return &y }())
					}
					if j < q {
						goblas.Daxpy(toPtr((*n)-je), rhs.GetPtr(1), b.Vector(js-1, je+1-1), ldb, c.Vector(is-1, je+1-1), ldc)
						goblas.Daxpy(toPtr((*n)-je), rhs.GetPtr(1), e.Vector(js-1, je+1-1), lde, f.Vector(is-1, je+1-1), ldf)
					}

				} else if (mb == 1) && (nb == 2) {
					//                 Build a 4-by-4 system Z * x = RHS
					z.Set(0, 0, a.Get(is-1, is-1))
					z.Set(1, 0, zero)
					z.Set(2, 0, d.Get(is-1, is-1))
					z.Set(3, 0, zero)

					z.Set(0, 1, zero)
					z.Set(1, 1, a.Get(is-1, is-1))
					z.Set(2, 1, zero)
					z.Set(3, 1, d.Get(is-1, is-1))

					z.Set(0, 2, -b.Get(js-1, js-1))
					z.Set(1, 2, -b.Get(js-1, jsp1-1))
					z.Set(2, 2, -e.Get(js-1, js-1))
					z.Set(3, 2, -e.Get(js-1, jsp1-1))

					z.Set(0, 3, -b.Get(jsp1-1, js-1))
					z.Set(1, 3, -b.Get(jsp1-1, jsp1-1))
					z.Set(2, 3, zero)
					z.Set(3, 3, -e.Get(jsp1-1, jsp1-1))

					//                 Set up right hand side(s)
					rhs.Set(0, c.Get(is-1, js-1))
					rhs.Set(1, c.Get(is-1, jsp1-1))
					rhs.Set(2, f.Get(is-1, js-1))
					rhs.Set(3, f.Get(is-1, jsp1-1))

					//                 Solve Z * x = RHS
					Dgetc2(&zdim, z, &ldz, &ipiv, &jpiv, &ierr)
					if ierr > 0 {
						(*info) = ierr
					}

					if (*ijob) == 0 {
						Dgesc2(&zdim, z, &ldz, rhs, &ipiv, &jpiv, &scaloc)
						if scaloc != one {
							for k = 1; k <= (*n); k++ {
								goblas.Dscal(m, &scaloc, c.Vector(0, k-1), func() *int { y := 1; return &y }())
								goblas.Dscal(m, &scaloc, f.Vector(0, k-1), func() *int { y := 1; return &y }())
							}
							(*scale) = (*scale) * scaloc
						}
					} else {
						Dlatdf(ijob, &zdim, z, &ldz, rhs, rdsum, rdscal, &ipiv, &jpiv)
					}

					//                 Unpack solution vector(s)
					c.Set(is-1, js-1, rhs.Get(0))
					c.Set(is-1, jsp1-1, rhs.Get(1))
					f.Set(is-1, js-1, rhs.Get(2))
					f.Set(is-1, jsp1-1, rhs.Get(3))

					//                 Substitute R(I, J) and L(I, J) into remaining
					//                 equation.
					if i > 1 {
						goblas.Dger(toPtr(is-1), &nb, toPtrf64(-one), a.Vector(0, is-1), func() *int { y := 1; return &y }(), rhs, func() *int { y := 1; return &y }(), c.Off(0, js-1), ldc)
						goblas.Dger(toPtr(is-1), &nb, toPtrf64(-one), d.Vector(0, is-1), func() *int { y := 1; return &y }(), rhs, func() *int { y := 1; return &y }(), f.Off(0, js-1), ldf)
					}
					if j < q {
						goblas.Daxpy(toPtr((*n)-je), rhs.GetPtr(2), b.Vector(js-1, je+1-1), ldb, c.Vector(is-1, je+1-1), ldc)
						goblas.Daxpy(toPtr((*n)-je), rhs.GetPtr(2), e.Vector(js-1, je+1-1), lde, f.Vector(is-1, je+1-1), ldf)
						goblas.Daxpy(toPtr((*n)-je), rhs.GetPtr(3), b.Vector(jsp1-1, je+1-1), ldb, c.Vector(is-1, je+1-1), ldc)
						goblas.Daxpy(toPtr((*n)-je), rhs.GetPtr(3), e.Vector(jsp1-1, je+1-1), lde, f.Vector(is-1, je+1-1), ldf)
					}

				} else if (mb == 2) && (nb == 1) {
					//                 Build a 4-by-4 system Z * x = RHS
					z.Set(0, 0, a.Get(is-1, is-1))
					z.Set(1, 0, a.Get(isp1-1, is-1))
					z.Set(2, 0, d.Get(is-1, is-1))
					z.Set(3, 0, zero)

					z.Set(0, 1, a.Get(is-1, isp1-1))
					z.Set(1, 1, a.Get(isp1-1, isp1-1))
					z.Set(2, 1, d.Get(is-1, isp1-1))
					z.Set(3, 1, d.Get(isp1-1, isp1-1))

					z.Set(0, 2, -b.Get(js-1, js-1))
					z.Set(1, 2, zero)
					z.Set(2, 2, -e.Get(js-1, js-1))
					z.Set(3, 2, zero)

					z.Set(0, 3, zero)
					z.Set(1, 3, -b.Get(js-1, js-1))
					z.Set(2, 3, zero)
					z.Set(3, 3, -e.Get(js-1, js-1))

					//                 Set up right hand side(s)
					rhs.Set(0, c.Get(is-1, js-1))
					rhs.Set(1, c.Get(isp1-1, js-1))
					rhs.Set(2, f.Get(is-1, js-1))
					rhs.Set(3, f.Get(isp1-1, js-1))

					//                 Solve Z * x = RHS
					Dgetc2(&zdim, z, &ldz, &ipiv, &jpiv, &ierr)
					if ierr > 0 {
						(*info) = ierr
					}
					if (*ijob) == 0 {
						Dgesc2(&zdim, z, &ldz, rhs, &ipiv, &jpiv, &scaloc)
						if scaloc != one {
							for k = 1; k <= (*n); k++ {
								goblas.Dscal(m, &scaloc, c.Vector(0, k-1), func() *int { y := 1; return &y }())
								goblas.Dscal(m, &scaloc, f.Vector(0, k-1), func() *int { y := 1; return &y }())
							}
							(*scale) = (*scale) * scaloc
						}
					} else {
						Dlatdf(ijob, &zdim, z, &ldz, rhs, rdsum, rdscal, &ipiv, &jpiv)
					}

					//                 Unpack solution vector(s)
					c.Set(is-1, js-1, rhs.Get(0))
					c.Set(isp1-1, js-1, rhs.Get(1))
					f.Set(is-1, js-1, rhs.Get(2))
					f.Set(isp1-1, js-1, rhs.Get(3))

					//                 Substitute R(I, J) and L(I, J) into remaining
					//                 equation.
					if i > 1 {
						goblas.Dgemv(NoTrans, toPtr(is-1), &mb, toPtrf64(-one), a.Off(0, is-1), lda, rhs.Off(0), func() *int { y := 1; return &y }(), &one, c.Vector(0, js-1), func() *int { y := 1; return &y }())
						goblas.Dgemv(NoTrans, toPtr(is-1), &mb, toPtrf64(-one), d.Off(0, is-1), ldd, rhs.Off(0), func() *int { y := 1; return &y }(), &one, f.Vector(0, js-1), func() *int { y := 1; return &y }())
					}
					if j < q {
						goblas.Dger(&mb, toPtr((*n)-je), &one, rhs.Off(2), func() *int { y := 1; return &y }(), b.Vector(js-1, je+1-1), ldb, c.Off(is-1, je+1-1), ldc)
						goblas.Dger(&mb, toPtr((*n)-je), &one, rhs.Off(2), func() *int { y := 1; return &y }(), e.Vector(js-1, je+1-1), lde, f.Off(is-1, je+1-1), ldf)
					}

				} else if (mb == 2) && (nb == 2) {
					//                 Build an 8-by-8 system Z * x = RHS
					Dlaset('F', &ldz, &ldz, &zero, &zero, z, &ldz)

					z.Set(0, 0, a.Get(is-1, is-1))
					z.Set(1, 0, a.Get(isp1-1, is-1))
					z.Set(4, 0, d.Get(is-1, is-1))

					z.Set(0, 1, a.Get(is-1, isp1-1))
					z.Set(1, 1, a.Get(isp1-1, isp1-1))
					z.Set(4, 1, d.Get(is-1, isp1-1))
					z.Set(5, 1, d.Get(isp1-1, isp1-1))

					z.Set(2, 2, a.Get(is-1, is-1))
					z.Set(3, 2, a.Get(isp1-1, is-1))
					z.Set(6, 2, d.Get(is-1, is-1))

					z.Set(2, 3, a.Get(is-1, isp1-1))
					z.Set(3, 3, a.Get(isp1-1, isp1-1))
					z.Set(6, 3, d.Get(is-1, isp1-1))
					z.Set(7, 3, d.Get(isp1-1, isp1-1))

					z.Set(0, 4, -b.Get(js-1, js-1))
					z.Set(2, 4, -b.Get(js-1, jsp1-1))
					z.Set(4, 4, -e.Get(js-1, js-1))
					z.Set(6, 4, -e.Get(js-1, jsp1-1))

					z.Set(1, 5, -b.Get(js-1, js-1))
					z.Set(3, 5, -b.Get(js-1, jsp1-1))
					z.Set(5, 5, -e.Get(js-1, js-1))
					z.Set(7, 5, -e.Get(js-1, jsp1-1))

					z.Set(0, 6, -b.Get(jsp1-1, js-1))
					z.Set(2, 6, -b.Get(jsp1-1, jsp1-1))
					z.Set(6, 6, -e.Get(jsp1-1, jsp1-1))

					z.Set(1, 7, -b.Get(jsp1-1, js-1))
					z.Set(3, 7, -b.Get(jsp1-1, jsp1-1))
					z.Set(7, 7, -e.Get(jsp1-1, jsp1-1))

					//                 Set up right hand side(s)
					k = 1
					ii = mb*nb + 1
					for jj = 0; jj <= nb-1; jj++ {
						goblas.Dcopy(&mb, c.Vector(is-1, js+jj-1), func() *int { y := 1; return &y }(), rhs.Off(k-1), func() *int { y := 1; return &y }())
						goblas.Dcopy(&mb, f.Vector(is-1, js+jj-1), func() *int { y := 1; return &y }(), rhs.Off(ii-1), func() *int { y := 1; return &y }())
						k = k + mb
						ii = ii + mb
					}

					//                 Solve Z * x = RHS
					Dgetc2(&zdim, z, &ldz, &ipiv, &jpiv, &ierr)
					if ierr > 0 {
						(*info) = ierr
					}
					if (*ijob) == 0 {
						Dgesc2(&zdim, z, &ldz, rhs, &ipiv, &jpiv, &scaloc)
						if scaloc != one {
							for k = 1; k <= (*n); k++ {
								goblas.Dscal(m, &scaloc, c.Vector(0, k-1), func() *int { y := 1; return &y }())
								goblas.Dscal(m, &scaloc, f.Vector(0, k-1), func() *int { y := 1; return &y }())
							}
							(*scale) = (*scale) * scaloc
						}
					} else {
						Dlatdf(ijob, &zdim, z, &ldz, rhs, rdsum, rdscal, &ipiv, &jpiv)
					}

					//                 Unpack solution vector(s)
					k = 1
					ii = mb*nb + 1
					for jj = 0; jj <= nb-1; jj++ {
						goblas.Dcopy(&mb, rhs.Off(k-1), func() *int { y := 1; return &y }(), c.Vector(is-1, js+jj-1), func() *int { y := 1; return &y }())
						goblas.Dcopy(&mb, rhs.Off(ii-1), func() *int { y := 1; return &y }(), f.Vector(is-1, js+jj-1), func() *int { y := 1; return &y }())
						k = k + mb
						ii = ii + mb
					}

					//                 Substitute R(I, J) and L(I, J) into remaining
					//                 equation.
					if i > 1 {
						goblas.Dgemm(NoTrans, NoTrans, toPtr(is-1), &nb, &mb, toPtrf64(-one), a.Off(0, is-1), lda, rhs.Matrix(mb, opts), &mb, &one, c.Off(0, js-1), ldc)
						goblas.Dgemm(NoTrans, NoTrans, toPtr(is-1), &nb, &mb, toPtrf64(-one), d.Off(0, is-1), ldd, rhs.Matrix(mb, opts), &mb, &one, f.Off(0, js-1), ldf)
					}
					if j < q {
						k = mb*nb + 1
						goblas.Dgemm(NoTrans, NoTrans, &mb, toPtr((*n)-je), &nb, &one, rhs.MatrixOff(k-1, mb, opts), &mb, b.Off(js-1, je+1-1), ldb, &one, c.Off(is-1, je+1-1), ldc)
						goblas.Dgemm(NoTrans, NoTrans, &mb, toPtr((*n)-je), &nb, &one, rhs.MatrixOff(k-1, mb, opts), &mb, e.Off(js-1, je+1-1), lde, &one, f.Off(is-1, je+1-1), ldf)
					}

				}

			}
		}
	} else {
		//        Solve (I, J) - subsystem
		//             A(I, I)**T * R(I, J) + D(I, I)**T * L(J, J)  =  C(I, J)
		//             R(I, I)  * B(J, J) + L(I, J)  * E(J, J)  = -F(I, J)
		//        for I = 1, 2, ..., P, J = Q, Q - 1, ..., 1
		(*scale) = one
		scaloc = one
		for i = 1; i <= p; i++ {

			is = (*iwork)[i-1]
			isp1 = is + 1
			ie = (*iwork)[i+1-1] - 1
			mb = ie - is + 1
			for j = q; j >= p+2; j-- {

				js = (*iwork)[j-1]
				jsp1 = js + 1
				je = (*iwork)[j+1-1] - 1
				nb = je - js + 1
				zdim = mb * nb * 2
				if (mb == 1) && (nb == 1) {
					//                 Build a 2-by-2 system Z**T * x = RHS
					z.Set(0, 0, a.Get(is-1, is-1))
					z.Set(1, 0, -b.Get(js-1, js-1))
					z.Set(0, 1, d.Get(is-1, is-1))
					z.Set(1, 1, -e.Get(js-1, js-1))

					//                 Set up right hand side(s)
					rhs.Set(0, c.Get(is-1, js-1))
					rhs.Set(1, f.Get(is-1, js-1))

					//                 Solve Z**T * x = RHS
					Dgetc2(&zdim, z, &ldz, &ipiv, &jpiv, &ierr)
					if ierr > 0 {
						(*info) = ierr
					}

					Dgesc2(&zdim, z, &ldz, rhs, &ipiv, &jpiv, &scaloc)
					if scaloc != one {
						for k = 1; k <= (*n); k++ {
							goblas.Dscal(m, &scaloc, c.Vector(0, k-1), func() *int { y := 1; return &y }())
							goblas.Dscal(m, &scaloc, f.Vector(0, k-1), func() *int { y := 1; return &y }())
						}
						(*scale) = (*scale) * scaloc
					}

					//                 Unpack solution vector(s)
					c.Set(is-1, js-1, rhs.Get(0))
					f.Set(is-1, js-1, rhs.Get(1))

					//                 Substitute R(I, J) and L(I, J) into remaining
					//                 equation.
					if j > p+2 {
						alpha = rhs.Get(0)
						goblas.Daxpy(toPtr(js-1), &alpha, b.Vector(0, js-1), func() *int { y := 1; return &y }(), f.Vector(is-1, 0), ldf)
						alpha = rhs.Get(1)
						goblas.Daxpy(toPtr(js-1), &alpha, e.Vector(0, js-1), func() *int { y := 1; return &y }(), f.Vector(is-1, 0), ldf)
					}
					if i < p {
						alpha = -rhs.Get(0)
						goblas.Daxpy(toPtr((*m)-ie), &alpha, a.Vector(is-1, ie+1-1), lda, c.Vector(ie+1-1, js-1), func() *int { y := 1; return &y }())
						alpha = -rhs.Get(1)
						goblas.Daxpy(toPtr((*m)-ie), &alpha, d.Vector(is-1, ie+1-1), ldd, c.Vector(ie+1-1, js-1), func() *int { y := 1; return &y }())
					}

				} else if (mb == 1) && (nb == 2) {
					//                 Build a 4-by-4 system Z**T * x = RHS
					z.Set(0, 0, a.Get(is-1, is-1))
					z.Set(1, 0, zero)
					z.Set(2, 0, -b.Get(js-1, js-1))
					z.Set(3, 0, -b.Get(jsp1-1, js-1))

					z.Set(0, 1, zero)
					z.Set(1, 1, a.Get(is-1, is-1))
					z.Set(2, 1, -b.Get(js-1, jsp1-1))
					z.Set(3, 1, -b.Get(jsp1-1, jsp1-1))

					z.Set(0, 2, d.Get(is-1, is-1))
					z.Set(1, 2, zero)
					z.Set(2, 2, -e.Get(js-1, js-1))
					z.Set(3, 2, zero)

					z.Set(0, 3, zero)
					z.Set(1, 3, d.Get(is-1, is-1))
					z.Set(2, 3, -e.Get(js-1, jsp1-1))
					z.Set(3, 3, -e.Get(jsp1-1, jsp1-1))

					//                 Set up right hand side(s)
					rhs.Set(0, c.Get(is-1, js-1))
					rhs.Set(1, c.Get(is-1, jsp1-1))
					rhs.Set(2, f.Get(is-1, js-1))
					rhs.Set(3, f.Get(is-1, jsp1-1))

					//                 Solve Z**T * x = RHS
					Dgetc2(&zdim, z, &ldz, &ipiv, &jpiv, &ierr)
					if ierr > 0 {
						(*info) = ierr
					}
					Dgesc2(&zdim, z, &ldz, rhs, &ipiv, &jpiv, &scaloc)
					if scaloc != one {
						for k = 1; k <= (*n); k++ {
							goblas.Dscal(m, &scaloc, c.Vector(0, k-1), func() *int { y := 1; return &y }())
							goblas.Dscal(m, &scaloc, f.Vector(0, k-1), func() *int { y := 1; return &y }())
						}
						(*scale) = (*scale) * scaloc
					}

					//                 Unpack solution vector(s)
					c.Set(is-1, js-1, rhs.Get(0))
					c.Set(is-1, jsp1-1, rhs.Get(1))
					f.Set(is-1, js-1, rhs.Get(2))
					f.Set(is-1, jsp1-1, rhs.Get(3))

					//                 Substitute R(I, J) and L(I, J) into remaining
					//                 equation.
					if j > p+2 {
						goblas.Daxpy(toPtr(js-1), rhs.GetPtr(0), b.Vector(0, js-1), func() *int { y := 1; return &y }(), f.Vector(is-1, 0), ldf)
						goblas.Daxpy(toPtr(js-1), rhs.GetPtr(1), b.Vector(0, jsp1-1), func() *int { y := 1; return &y }(), f.Vector(is-1, 0), ldf)
						goblas.Daxpy(toPtr(js-1), rhs.GetPtr(2), e.Vector(0, js-1), func() *int { y := 1; return &y }(), f.Vector(is-1, 0), ldf)
						goblas.Daxpy(toPtr(js-1), rhs.GetPtr(3), e.Vector(0, jsp1-1), func() *int { y := 1; return &y }(), f.Vector(is-1, 0), ldf)
					}
					if i < p {
						goblas.Dger(toPtr((*m)-ie), &nb, toPtrf64(-one), a.Vector(is-1, ie+1-1), lda, rhs.Off(0), func() *int { y := 1; return &y }(), c.Off(ie+1-1, js-1), ldc)
						goblas.Dger(toPtr((*m)-ie), &nb, toPtrf64(-one), d.Vector(is-1, ie+1-1), ldd, rhs.Off(2), func() *int { y := 1; return &y }(), c.Off(ie+1-1, js-1), ldc)
					}

				} else if (mb == 2) && (nb == 1) {
					//                 Build a 4-by-4 system Z**T * x = RHS
					z.Set(0, 0, a.Get(is-1, is-1))
					z.Set(1, 0, a.Get(is-1, isp1-1))
					z.Set(2, 0, -b.Get(js-1, js-1))
					z.Set(3, 0, zero)

					z.Set(0, 1, a.Get(isp1-1, is-1))
					z.Set(1, 1, a.Get(isp1-1, isp1-1))
					z.Set(2, 1, zero)
					z.Set(3, 1, -b.Get(js-1, js-1))

					z.Set(0, 2, d.Get(is-1, is-1))
					z.Set(1, 2, d.Get(is-1, isp1-1))
					z.Set(2, 2, -e.Get(js-1, js-1))
					z.Set(3, 2, zero)

					z.Set(0, 3, zero)
					z.Set(1, 3, d.Get(isp1-1, isp1-1))
					z.Set(2, 3, zero)
					z.Set(3, 3, -e.Get(js-1, js-1))

					//                 Set up right hand side(s)
					rhs.Set(0, c.Get(is-1, js-1))
					rhs.Set(1, c.Get(isp1-1, js-1))
					rhs.Set(2, f.Get(is-1, js-1))
					rhs.Set(3, f.Get(isp1-1, js-1))

					//                 Solve Z**T * x = RHS
					Dgetc2(&zdim, z, &ldz, &ipiv, &jpiv, &ierr)
					if ierr > 0 {
						(*info) = ierr
					}

					Dgesc2(&zdim, z, &ldz, rhs, &ipiv, &jpiv, &scaloc)
					if scaloc != one {
						for k = 1; k <= (*n); k++ {
							goblas.Dscal(m, &scaloc, c.Vector(0, k-1), func() *int { y := 1; return &y }())
							goblas.Dscal(m, &scaloc, f.Vector(0, k-1), func() *int { y := 1; return &y }())
						}
						(*scale) = (*scale) * scaloc
					}

					//                 Unpack solution vector(s)
					c.Set(is-1, js-1, rhs.Get(0))
					c.Set(isp1-1, js-1, rhs.Get(1))
					f.Set(is-1, js-1, rhs.Get(2))
					f.Set(isp1-1, js-1, rhs.Get(3))

					//                 Substitute R(I, J) and L(I, J) into remaining
					//                 equation.
					if j > p+2 {
						goblas.Dger(&mb, toPtr(js-1), &one, rhs.Off(0), func() *int { y := 1; return &y }(), b.Vector(0, js-1), func() *int { y := 1; return &y }(), f.Off(is-1, 0), ldf)
						goblas.Dger(&mb, toPtr(js-1), &one, rhs.Off(2), func() *int { y := 1; return &y }(), e.Vector(0, js-1), func() *int { y := 1; return &y }(), f.Off(is-1, 0), ldf)
					}
					if i < p {
						goblas.Dgemv(Trans, &mb, toPtr((*m)-ie), toPtrf64(-one), a.Off(is-1, ie+1-1), lda, rhs.Off(0), func() *int { y := 1; return &y }(), &one, c.Vector(ie+1-1, js-1), func() *int { y := 1; return &y }())
						goblas.Dgemv(Trans, &mb, toPtr((*m)-ie), toPtrf64(-one), d.Off(is-1, ie+1-1), ldd, rhs.Off(2), func() *int { y := 1; return &y }(), &one, c.Vector(ie+1-1, js-1), func() *int { y := 1; return &y }())
					}

				} else if (mb == 2) && (nb == 2) {
					//                 Build an 8-by-8 system Z**T * x = RHS
					Dlaset('F', &ldz, &ldz, &zero, &zero, z, &ldz)

					z.Set(0, 0, a.Get(is-1, is-1))
					z.Set(1, 0, a.Get(is-1, isp1-1))
					z.Set(4, 0, -b.Get(js-1, js-1))
					z.Set(6, 0, -b.Get(jsp1-1, js-1))

					z.Set(0, 1, a.Get(isp1-1, is-1))
					z.Set(1, 1, a.Get(isp1-1, isp1-1))
					z.Set(5, 1, -b.Get(js-1, js-1))
					z.Set(7, 1, -b.Get(jsp1-1, js-1))

					z.Set(2, 2, a.Get(is-1, is-1))
					z.Set(3, 2, a.Get(is-1, isp1-1))
					z.Set(4, 2, -b.Get(js-1, jsp1-1))
					z.Set(6, 2, -b.Get(jsp1-1, jsp1-1))

					z.Set(2, 3, a.Get(isp1-1, is-1))
					z.Set(3, 3, a.Get(isp1-1, isp1-1))
					z.Set(5, 3, -b.Get(js-1, jsp1-1))
					z.Set(7, 3, -b.Get(jsp1-1, jsp1-1))

					z.Set(0, 4, d.Get(is-1, is-1))
					z.Set(1, 4, d.Get(is-1, isp1-1))
					z.Set(4, 4, -e.Get(js-1, js-1))

					z.Set(1, 5, d.Get(isp1-1, isp1-1))
					z.Set(5, 5, -e.Get(js-1, js-1))

					z.Set(2, 6, d.Get(is-1, is-1))
					z.Set(3, 6, d.Get(is-1, isp1-1))
					z.Set(4, 6, -e.Get(js-1, jsp1-1))
					z.Set(6, 6, -e.Get(jsp1-1, jsp1-1))

					z.Set(3, 7, d.Get(isp1-1, isp1-1))
					z.Set(5, 7, -e.Get(js-1, jsp1-1))
					z.Set(7, 7, -e.Get(jsp1-1, jsp1-1))

					//                 Set up right hand side(s)
					k = 1
					ii = mb*nb + 1
					for jj = 0; jj <= nb-1; jj++ {
						goblas.Dcopy(&mb, c.Vector(is-1, js+jj-1), func() *int { y := 1; return &y }(), rhs.Off(k-1), func() *int { y := 1; return &y }())
						goblas.Dcopy(&mb, f.Vector(is-1, js+jj-1), func() *int { y := 1; return &y }(), rhs.Off(ii-1), func() *int { y := 1; return &y }())
						k = k + mb
						ii = ii + mb
					}

					//                 Solve Z**T * x = RHS
					Dgetc2(&zdim, z, &ldz, &ipiv, &jpiv, &ierr)
					if ierr > 0 {
						(*info) = ierr
					}

					Dgesc2(&zdim, z, &ldz, rhs, &ipiv, &jpiv, &scaloc)
					if scaloc != one {
						for k = 1; k <= (*n); k++ {
							goblas.Dscal(m, &scaloc, c.Vector(0, k-1), func() *int { y := 1; return &y }())
							goblas.Dscal(m, &scaloc, f.Vector(0, k-1), func() *int { y := 1; return &y }())
						}
						(*scale) = (*scale) * scaloc
					}

					//                 Unpack solution vector(s)
					k = 1
					ii = mb*nb + 1
					for jj = 0; jj <= nb-1; jj++ {
						goblas.Dcopy(&mb, rhs.Off(k-1), func() *int { y := 1; return &y }(), c.Vector(is-1, js+jj-1), func() *int { y := 1; return &y }())
						goblas.Dcopy(&mb, rhs.Off(ii-1), func() *int { y := 1; return &y }(), f.Vector(is-1, js+jj-1), func() *int { y := 1; return &y }())
						k = k + mb
						ii = ii + mb
					}

					//                 Substitute R(I, J) and L(I, J) into remaining
					//                 equation.
					if j > p+2 {
						goblas.Dgemm(NoTrans, Trans, &mb, toPtr(js-1), &nb, &one, c.Off(is-1, js-1), ldc, b.Off(0, js-1), ldb, &one, f.Off(is-1, 0), ldf)
						goblas.Dgemm(NoTrans, Trans, &mb, toPtr(js-1), &nb, &one, f.Off(is-1, js-1), ldf, e.Off(0, js-1), lde, &one, f.Off(is-1, 0), ldf)
					}
					if i < p {
						goblas.Dgemm(Trans, NoTrans, toPtr((*m)-ie), &nb, &mb, toPtrf64(-one), a.Off(is-1, ie+1-1), lda, c.Off(is-1, js-1), ldc, &one, c.Off(ie+1-1, js-1), ldc)
						goblas.Dgemm(Trans, NoTrans, toPtr((*m)-ie), &nb, &mb, toPtrf64(-one), d.Off(is-1, ie+1-1), ldd, f.Off(is-1, js-1), ldf, &one, c.Off(ie+1-1, js-1), ldc)
					}

				}

			}
		}

	}
}
