package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
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
// Dtgsy2 also (IJOB >= 1) contributes to the computation in DTGSYL
// of an upper bound on the separation between to matrix pairs. Then
// the input (A, D), (B, E) are sub-pencils of the matrix pair in
// DTGSYL. See DTGSYL for details.
func Dtgsy2(trans mat.MatTrans, ijob, m, n int, a, b, c, d, e, f *mat.Matrix, rdsum, rdscal float64, iwork *[]int) (scale, rdsumOut, rdscalOut float64, pq, info int, err error) {
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
	rdsumOut = rdsum
	rdscalOut = rdscal

	//     Decode and test input parameters
	ierr = 0
	notran = trans == NoTrans
	if !notran && trans != Trans {
		err = fmt.Errorf("!notran && trans != Trans: trans=%s", trans)
	} else if notran {
		if (ijob < 0) || (ijob > 2) {
			err = fmt.Errorf("(ijob < 0) || (ijob > 2): ijob=%v", ijob)
		}
	}
	if err == nil {
		if m <= 0 {
			err = fmt.Errorf("m <= 0: m=%v", m)
		} else if n <= 0 {
			err = fmt.Errorf("n <= 0: n=%v", n)
		} else if a.Rows < max(1, m) {
			err = fmt.Errorf("a.Rows < max(1, m): a.Rows=%v, m=%v", a.Rows, m)
		} else if b.Rows < max(1, n) {
			err = fmt.Errorf("b.Rows < max(1, n): b.Rows=%v, n=%v", b.Rows, n)
		} else if c.Rows < max(1, m) {
			err = fmt.Errorf("c.Rows < max(1, m): c.Rows=%v, m=%v", c.Rows, m)
		} else if d.Rows < max(1, m) {
			err = fmt.Errorf("d.Rows < max(1, m): d.Rows=%v, m=%v", d.Rows, m)
		} else if e.Rows < max(1, n) {
			err = fmt.Errorf("e.Rows < max(1, n): e.Rows=%v, n=%v", e.Rows, n)
		} else if f.Rows < max(1, m) {
			err = fmt.Errorf("f.Rows < max(1, m): f.Rows=%v, m=%v", f.Rows, m)
		}
	}
	if err != nil {
		gltest.Xerbla2("Dtgsy2", err)
		return
	}

	//     Determine block structure of A
	pq = 0
	p = 0
	i = 1
label10:
	;
	if i > m {
		goto label20
	}
	p = p + 1
	(*iwork)[p-1] = i
	if i == m {
		goto label20
	}
	if a.Get(i, i-1) != zero {
		i = i + 2
	} else {
		i = i + 1
	}
	goto label10
label20:
	;
	(*iwork)[p] = m + 1

	//     Determine block structure of B
	q = p + 1
	j = 1
label30:
	;
	if j > n {
		goto label40
	}
	q = q + 1
	(*iwork)[q-1] = j
	if j == n {
		goto label40
	}
	if b.Get(j, j-1) != zero {
		j = j + 2
	} else {
		j = j + 1
	}
	goto label30
label40:
	;
	(*iwork)[q] = n + 1
	pq = p * (q - p - 1)

	if notran {
		//        Solve (I, J) - subsystem
		//           A(I, I) * R(I, J) - L(I, J) * B(J, J) = C(I, J)
		//           D(I, I) * R(I, J) - L(I, J) * E(J, J) = F(I, J)
		//        for I = P, P - 1, ..., 1; J = 1, 2, ..., Q
		scale = one
		scaloc = one
		for j = p + 2; j <= q; j++ {
			js = (*iwork)[j-1]
			jsp1 = js + 1
			je = (*iwork)[j] - 1
			nb = je - js + 1
			for i = p; i >= 1; i-- {

				is = (*iwork)[i-1]
				isp1 = is + 1
				ie = (*iwork)[i] - 1
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
					if ierr = Dgetc2(zdim, z, &ipiv, &jpiv); ierr > 0 {
						info = ierr
					}

					if ijob == 0 {
						scaloc = Dgesc2(zdim, z, rhs, &ipiv, &jpiv)
						if scaloc != one {
							for k = 1; k <= n; k++ {
								goblas.Dscal(m, scaloc, c.Vector(0, k-1, 1))
								goblas.Dscal(m, scaloc, f.Vector(0, k-1, 1))
							}
							scale = scale * scaloc
						}
					} else {
						rdsumOut, rdscalOut = Dlatdf(ijob, zdim, z, rhs, rdsumOut, rdscalOut, &ipiv, &jpiv)
					}

					//                 Unpack solution vector(s)
					c.Set(is-1, js-1, rhs.Get(0))
					f.Set(is-1, js-1, rhs.Get(1))

					//                 Substitute R(I, J) and L(I, J) into remaining
					//                 equation.
					if i > 1 {
						alpha = -rhs.Get(0)
						goblas.Daxpy(is-1, alpha, a.Vector(0, is-1, 1), c.Vector(0, js-1, 1))
						goblas.Daxpy(is-1, alpha, d.Vector(0, is-1, 1), f.Vector(0, js-1, 1))
					}
					if j < q {
						goblas.Daxpy(n-je, rhs.Get(1), b.Vector(js-1, je), c.Vector(is-1, je))
						goblas.Daxpy(n-je, rhs.Get(1), e.Vector(js-1, je), f.Vector(is-1, je))
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
					if ierr = Dgetc2(zdim, z, &ipiv, &jpiv); ierr > 0 {
						info = ierr
					}

					if ijob == 0 {
						scaloc = Dgesc2(zdim, z, rhs, &ipiv, &jpiv)
						if scaloc != one {
							for k = 1; k <= n; k++ {
								goblas.Dscal(m, scaloc, c.Vector(0, k-1, 1))
								goblas.Dscal(m, scaloc, f.Vector(0, k-1, 1))
							}
							scale = scale * scaloc
						}
					} else {
						rdsumOut, rdscalOut = Dlatdf(ijob, zdim, z, rhs, rdsumOut, rdscalOut, &ipiv, &jpiv)
					}

					//                 Unpack solution vector(s)
					c.Set(is-1, js-1, rhs.Get(0))
					c.Set(is-1, jsp1-1, rhs.Get(1))
					f.Set(is-1, js-1, rhs.Get(2))
					f.Set(is-1, jsp1-1, rhs.Get(3))

					//                 Substitute R(I, J) and L(I, J) into remaining
					//                 equation.
					if i > 1 {
						if err = goblas.Dger(is-1, nb, -one, a.Vector(0, is-1, 1), rhs.Off(0, 1), c.Off(0, js-1)); err != nil {
							panic(err)
						}
						if err = goblas.Dger(is-1, nb, -one, d.Vector(0, is-1, 1), rhs.Off(0, 1), f.Off(0, js-1)); err != nil {
							panic(err)
						}
					}
					if j < q {
						goblas.Daxpy(n-je, rhs.Get(2), b.Vector(js-1, je), c.Vector(is-1, je))
						goblas.Daxpy(n-je, rhs.Get(2), e.Vector(js-1, je), f.Vector(is-1, je))
						goblas.Daxpy(n-je, rhs.Get(3), b.Vector(jsp1-1, je), c.Vector(is-1, je))
						goblas.Daxpy(n-je, rhs.Get(3), e.Vector(jsp1-1, je), f.Vector(is-1, je))
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
					if ierr = Dgetc2(zdim, z, &ipiv, &jpiv); ierr > 0 {
						info = ierr
					}
					if ijob == 0 {
						scaloc = Dgesc2(zdim, z, rhs, &ipiv, &jpiv)
						if scaloc != one {
							for k = 1; k <= n; k++ {
								goblas.Dscal(m, scaloc, c.Vector(0, k-1, 1))
								goblas.Dscal(m, scaloc, f.Vector(0, k-1, 1))
							}
							scale = scale * scaloc
						}
					} else {
						rdsumOut, rdscalOut = Dlatdf(ijob, zdim, z, rhs, rdsumOut, rdscalOut, &ipiv, &jpiv)
					}

					//                 Unpack solution vector(s)
					c.Set(is-1, js-1, rhs.Get(0))
					c.Set(isp1-1, js-1, rhs.Get(1))
					f.Set(is-1, js-1, rhs.Get(2))
					f.Set(isp1-1, js-1, rhs.Get(3))

					//                 Substitute R(I, J) and L(I, J) into remaining
					//                 equation.
					if i > 1 {
						if err = goblas.Dgemv(NoTrans, is-1, mb, -one, a.Off(0, is-1), rhs.Off(0, 1), one, c.Vector(0, js-1, 1)); err != nil {
							panic(err)
						}
						if err = goblas.Dgemv(NoTrans, is-1, mb, -one, d.Off(0, is-1), rhs.Off(0, 1), one, f.Vector(0, js-1, 1)); err != nil {
							panic(err)
						}
					}
					if j < q {
						if err = goblas.Dger(mb, n-je, one, rhs.Off(2, 1), b.Vector(js-1, je), c.Off(is-1, je)); err != nil {
							panic(err)
						}
						if err = goblas.Dger(mb, n-je, one, rhs.Off(2, 1), e.Vector(js-1, je), f.Off(is-1, je)); err != nil {
							panic(err)
						}
					}

				} else if (mb == 2) && (nb == 2) {
					//                 Build an 8-by-8 system Z * x = RHS
					Dlaset(Full, ldz, ldz, zero, zero, z)

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
						goblas.Dcopy(mb, c.Vector(is-1, js+jj-1, 1), rhs.Off(k-1, 1))
						goblas.Dcopy(mb, f.Vector(is-1, js+jj-1, 1), rhs.Off(ii-1, 1))
						k = k + mb
						ii = ii + mb
					}

					//                 Solve Z * x = RHS
					if ierr = Dgetc2(zdim, z, &ipiv, &jpiv); ierr > 0 {
						info = ierr
					}
					if ijob == 0 {
						scaloc = Dgesc2(zdim, z, rhs, &ipiv, &jpiv)
						if scaloc != one {
							for k = 1; k <= n; k++ {
								goblas.Dscal(m, scaloc, c.Vector(0, k-1, 1))
								goblas.Dscal(m, scaloc, f.Vector(0, k-1, 1))
							}
							scale = scale * scaloc
						}
					} else {
						rdsumOut, rdscalOut = Dlatdf(ijob, zdim, z, rhs, rdsumOut, rdscalOut, &ipiv, &jpiv)
					}

					//                 Unpack solution vector(s)
					k = 1
					ii = mb*nb + 1
					for jj = 0; jj <= nb-1; jj++ {
						goblas.Dcopy(mb, rhs.Off(k-1, 1), c.Vector(is-1, js+jj-1, 1))
						goblas.Dcopy(mb, rhs.Off(ii-1, 1), f.Vector(is-1, js+jj-1, 1))
						k = k + mb
						ii = ii + mb
					}

					//                 Substitute R(I, J) and L(I, J) into remaining
					//                 equation.
					if i > 1 {
						if err = goblas.Dgemm(NoTrans, NoTrans, is-1, nb, mb, -one, a.Off(0, is-1), rhs.Matrix(mb, opts), one, c.Off(0, js-1)); err != nil {
							panic(err)
						}
						if err = goblas.Dgemm(NoTrans, NoTrans, is-1, nb, mb, -one, d.Off(0, is-1), rhs.Matrix(mb, opts), one, f.Off(0, js-1)); err != nil {
							panic(err)
						}
					}
					if j < q {
						k = mb*nb + 1
						if err = goblas.Dgemm(NoTrans, NoTrans, mb, n-je, nb, one, rhs.MatrixOff(k-1, mb, opts), b.Off(js-1, je), one, c.Off(is-1, je)); err != nil {
							panic(err)
						}
						if err = goblas.Dgemm(NoTrans, NoTrans, mb, n-je, nb, one, rhs.MatrixOff(k-1, mb, opts), e.Off(js-1, je), one, f.Off(is-1, je)); err != nil {
							panic(err)
						}
					}

				}

			}
		}
	} else {
		//        Solve (I, J) - subsystem
		//             A(I, I)**T * R(I, J) + D(I, I)**T * L(J, J)  =  C(I, J)
		//             R(I, I)  * B(J, J) + L(I, J)  * E(J, J)  = -F(I, J)
		//        for I = 1, 2, ..., P, J = Q, Q - 1, ..., 1
		scale = one
		scaloc = one
		for i = 1; i <= p; i++ {

			is = (*iwork)[i-1]
			isp1 = is + 1
			ie = (*iwork)[i] - 1
			mb = ie - is + 1
			for j = q; j >= p+2; j-- {

				js = (*iwork)[j-1]
				jsp1 = js + 1
				je = (*iwork)[j] - 1
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
					if ierr = Dgetc2(zdim, z, &ipiv, &jpiv); ierr > 0 {
						info = ierr
					}

					scaloc = Dgesc2(zdim, z, rhs, &ipiv, &jpiv)
					if scaloc != one {
						for k = 1; k <= n; k++ {
							goblas.Dscal(m, scaloc, c.Vector(0, k-1, 1))
							goblas.Dscal(m, scaloc, f.Vector(0, k-1, 1))
						}
						scale = scale * scaloc
					}

					//                 Unpack solution vector(s)
					c.Set(is-1, js-1, rhs.Get(0))
					f.Set(is-1, js-1, rhs.Get(1))

					//                 Substitute R(I, J) and L(I, J) into remaining
					//                 equation.
					if j > p+2 {
						alpha = rhs.Get(0)
						goblas.Daxpy(js-1, alpha, b.Vector(0, js-1, 1), f.Vector(is-1, 0))
						alpha = rhs.Get(1)
						goblas.Daxpy(js-1, alpha, e.Vector(0, js-1, 1), f.Vector(is-1, 0))
					}
					if i < p {
						alpha = -rhs.Get(0)
						goblas.Daxpy(m-ie, alpha, a.Vector(is-1, ie), c.Vector(ie, js-1, 1))
						alpha = -rhs.Get(1)
						goblas.Daxpy(m-ie, alpha, d.Vector(is-1, ie), c.Vector(ie, js-1, 1))
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
					if ierr = Dgetc2(zdim, z, &ipiv, &jpiv); ierr > 0 {
						info = ierr
					}
					scaloc = Dgesc2(zdim, z, rhs, &ipiv, &jpiv)
					if scaloc != one {
						for k = 1; k <= n; k++ {
							goblas.Dscal(m, scaloc, c.Vector(0, k-1, 1))
							goblas.Dscal(m, scaloc, f.Vector(0, k-1, 1))
						}
						scale = scale * scaloc
					}

					//                 Unpack solution vector(s)
					c.Set(is-1, js-1, rhs.Get(0))
					c.Set(is-1, jsp1-1, rhs.Get(1))
					f.Set(is-1, js-1, rhs.Get(2))
					f.Set(is-1, jsp1-1, rhs.Get(3))

					//                 Substitute R(I, J) and L(I, J) into remaining
					//                 equation.
					if j > p+2 {
						goblas.Daxpy(js-1, rhs.Get(0), b.Vector(0, js-1, 1), f.Vector(is-1, 0))
						goblas.Daxpy(js-1, rhs.Get(1), b.Vector(0, jsp1-1, 1), f.Vector(is-1, 0))
						goblas.Daxpy(js-1, rhs.Get(2), e.Vector(0, js-1, 1), f.Vector(is-1, 0))
						goblas.Daxpy(js-1, rhs.Get(3), e.Vector(0, jsp1-1, 1), f.Vector(is-1, 0))
					}
					if i < p {
						if err = goblas.Dger(m-ie, nb, -one, a.Vector(is-1, ie), rhs.Off(0, 1), c.Off(ie, js-1)); err != nil {
							panic(err)
						}
						if err = goblas.Dger(m-ie, nb, -one, d.Vector(is-1, ie), rhs.Off(2, 1), c.Off(ie, js-1)); err != nil {
							panic(err)
						}
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
					if ierr = Dgetc2(zdim, z, &ipiv, &jpiv); ierr > 0 {
						info = ierr
					}

					scaloc = Dgesc2(zdim, z, rhs, &ipiv, &jpiv)
					if scaloc != one {
						for k = 1; k <= n; k++ {
							goblas.Dscal(m, scaloc, c.Vector(0, k-1, 1))
							goblas.Dscal(m, scaloc, f.Vector(0, k-1, 1))
						}
						scale = scale * scaloc
					}

					//                 Unpack solution vector(s)
					c.Set(is-1, js-1, rhs.Get(0))
					c.Set(isp1-1, js-1, rhs.Get(1))
					f.Set(is-1, js-1, rhs.Get(2))
					f.Set(isp1-1, js-1, rhs.Get(3))

					//                 Substitute R(I, J) and L(I, J) into remaining
					//                 equation.
					if j > p+2 {
						if err = goblas.Dger(mb, js-1, one, rhs.Off(0, 1), b.Vector(0, js-1, 1), f.Off(is-1, 0)); err != nil {
							panic(err)
						}
						if err = goblas.Dger(mb, js-1, one, rhs.Off(2, 1), e.Vector(0, js-1, 1), f.Off(is-1, 0)); err != nil {
							panic(err)
						}
					}
					if i < p {
						if err = goblas.Dgemv(Trans, mb, m-ie, -one, a.Off(is-1, ie), rhs.Off(0, 1), one, c.Vector(ie, js-1, 1)); err != nil {
							panic(err)
						}
						if err = goblas.Dgemv(Trans, mb, m-ie, -one, d.Off(is-1, ie), rhs.Off(2, 1), one, c.Vector(ie, js-1, 1)); err != nil {
							panic(err)
						}
					}

				} else if (mb == 2) && (nb == 2) {
					//                 Build an 8-by-8 system Z**T * x = RHS
					Dlaset(Full, ldz, ldz, zero, zero, z)

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
						goblas.Dcopy(mb, c.Vector(is-1, js+jj-1, 1), rhs.Off(k-1, 1))
						goblas.Dcopy(mb, f.Vector(is-1, js+jj-1, 1), rhs.Off(ii-1, 1))
						k = k + mb
						ii = ii + mb
					}

					//                 Solve Z**T * x = RHS
					if ierr = Dgetc2(zdim, z, &ipiv, &jpiv); ierr > 0 {
						info = ierr
					}

					scaloc = Dgesc2(zdim, z, rhs, &ipiv, &jpiv)
					if scaloc != one {
						for k = 1; k <= n; k++ {
							goblas.Dscal(m, scaloc, c.Vector(0, k-1, 1))
							goblas.Dscal(m, scaloc, f.Vector(0, k-1, 1))
						}
						scale = scale * scaloc
					}

					//                 Unpack solution vector(s)
					k = 1
					ii = mb*nb + 1
					for jj = 0; jj <= nb-1; jj++ {
						goblas.Dcopy(mb, rhs.Off(k-1, 1), c.Vector(is-1, js+jj-1, 1))
						goblas.Dcopy(mb, rhs.Off(ii-1, 1), f.Vector(is-1, js+jj-1, 1))
						k = k + mb
						ii = ii + mb
					}

					//                 Substitute R(I, J) and L(I, J) into remaining
					//                 equation.
					if j > p+2 {
						if err = goblas.Dgemm(NoTrans, Trans, mb, js-1, nb, one, c.Off(is-1, js-1), b.Off(0, js-1), one, f.Off(is-1, 0)); err != nil {
							panic(err)
						}
						if err = goblas.Dgemm(NoTrans, Trans, mb, js-1, nb, one, f.Off(is-1, js-1), e.Off(0, js-1), one, f.Off(is-1, 0)); err != nil {
							panic(err)
						}
					}
					if i < p {
						if err = goblas.Dgemm(Trans, NoTrans, m-ie, nb, mb, -one, a.Off(is-1, ie), c.Off(is-1, js-1), one, c.Off(ie, js-1)); err != nil {
							panic(err)
						}
						if err = goblas.Dgemm(Trans, NoTrans, m-ie, nb, mb, -one, d.Off(is-1, ie), f.Off(is-1, js-1), one, c.Off(ie, js-1)); err != nil {
							panic(err)
						}
					}

				}

			}
		}

	}

	return
}
