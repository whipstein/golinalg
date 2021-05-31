package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/mat"
)

// Dlaqtr solves the real quasi-triangular system
//
//              op(T)*p = scale*c,               if LREAL = .TRUE.
//
// or the complex quasi-triangular systems
//
//            op(T + iB)*(p+iq) = scale*(c+id),  if LREAL = .FALSE.
//
// in real arithmetic, where T is upper quasi-triangular.
// If LREAL = .FALSE., then the first diagonal block of T must be
// 1 by 1, B is the specially structured matrix
//
//                B = [ b(1) b(2) ... b(n) ]
//                    [       w            ]
//                    [           w        ]
//                    [              .     ]
//                    [                 w  ]
//
// op(A) = A or A**T, A**T denotes the transpose of
// matrix A.
//
// On input, X = [ c ].  On output, X = [ p ].
//               [ d ]                  [ q ]
//
// This subroutine is designed for the condition number estimation
// in routine DTRSNA.
func Dlaqtr(ltran, lreal bool, n *int, t *mat.Matrix, ldt *int, b *mat.Vector, w, scale *float64, x, work *mat.Vector, info *int) {
	var notran bool
	var bignum, eps, one, rec, scaloc, si, smin, sminw, smlnum, sr, tjj, tmp, xj, xmax, xnorm, z, zero float64
	var i, ierr, j, j1, j2, jnext, k, n1, n2 int

	d := mf(2, 2, opts)
	v := mf(2, 2, opts)

	zero = 0.0
	one = 1.0

	//     Do not test the input parameters for errors
	notran = !ltran
	(*info) = 0

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	//     Set constants to control overflow
	eps = Dlamch(Precision)
	smlnum = Dlamch(SafeMinimum) / eps
	bignum = one / smlnum

	xnorm = Dlange('M', n, n, t, ldt, d.VectorIdx(0))
	if !lreal {
		xnorm = maxf64(xnorm, math.Abs(*w), Dlange('M', n, func() *int { y := 1; return &y }(), b.Matrix(*n, opts), n, d.VectorIdx(0)))
	}
	smin = maxf64(smlnum, eps*xnorm)

	//     Compute 1-norm of each column of strictly upper triangular
	//     part of T to control overflow in triangular solver.
	work.Set(0, zero)
	for j = 2; j <= (*n); j++ {
		work.Set(j-1, goblas.Dasum(toPtr(j-1), t.Vector(0, j-1), func() *int { y := 1; return &y }()))
	}

	if !lreal {
		for i = 2; i <= (*n); i++ {
			work.Set(i-1, work.Get(i-1)+math.Abs(b.Get(i-1)))
		}
	}

	n2 = 2 * (*n)
	n1 = (*n)
	if !lreal {
		n1 = n2
	}
	k = goblas.Idamax(&n1, x, func() *int { y := 1; return &y }())
	xmax = math.Abs(x.Get(k - 1))
	(*scale) = one

	if xmax > bignum {
		(*scale) = bignum / xmax
		goblas.Dscal(&n1, scale, x, func() *int { y := 1; return &y }())
		xmax = bignum
	}

	if lreal {

		if notran {
			//           Solve T*p = scale*c
			jnext = (*n)
			for j = (*n); j >= 1; j-- {
				if j > jnext {
					goto label30
				}
				j1 = j
				j2 = j
				jnext = j - 1
				if j > 1 {
					if t.Get(j-1, j-1-1) != zero {
						j1 = j - 1
						jnext = j - 2
					}
				}

				if j1 == j2 {
					//                 Meet 1 by 1 diagonal block
					//
					//                 Scale to avoid overflow when computing
					//                     x(j) = b(j)/T(j,j)
					xj = math.Abs(x.Get(j1 - 1))
					tjj = math.Abs(t.Get(j1-1, j1-1))
					tmp = t.Get(j1-1, j1-1)
					if tjj < smin {
						tmp = smin
						tjj = smin
						(*info) = 1
					}

					if xj == zero {
						goto label30
					}

					if tjj < one {
						if xj > bignum*tjj {
							rec = one / xj
							goblas.Dscal(n, &rec, x, func() *int { y := 1; return &y }())
							(*scale) = (*scale) * rec
							xmax = xmax * rec
						}
					}
					x.Set(j1-1, x.Get(j1-1)/tmp)
					xj = math.Abs(x.Get(j1 - 1))

					//                 Scale x if necessary to avoid overflow when adding a
					//                 multiple of column j1 of T.
					if xj > one {
						rec = one / xj
						if work.Get(j1-1) > (bignum-xmax)*rec {
							goblas.Dscal(n, &rec, x, func() *int { y := 1; return &y }())
							(*scale) = (*scale) * rec
						}
					}
					if j1 > 1 {
						goblas.Daxpy(toPtr(j1-1), toPtrf64(-x.Get(j1-1)), t.Vector(0, j1-1), func() *int { y := 1; return &y }(), x, func() *int { y := 1; return &y }())
						k = goblas.Idamax(toPtr(j1-1), x, func() *int { y := 1; return &y }())
						xmax = math.Abs(x.Get(k - 1))
					}

				} else {
					//                 Meet 2 by 2 diagonal block
					//
					//                 Call 2 by 2 linear system solve, to take
					//                 care of possible overflow by scaling factor.
					d.Set(0, 0, x.Get(j1-1))
					d.Set(1, 0, x.Get(j2-1))
					Dlaln2(false, func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), &smin, &one, t.Off(j1-1, j1-1), ldt, &one, &one, d, func() *int { y := 2; return &y }(), &zero, &zero, v, func() *int { y := 2; return &y }(), &scaloc, &xnorm, &ierr)
					if ierr != 0 {
						(*info) = 2
					}

					if scaloc != one {
						goblas.Dscal(n, &scaloc, x, func() *int { y := 1; return &y }())
						(*scale) = (*scale) * scaloc
					}
					x.Set(j1-1, v.Get(0, 0))
					x.Set(j2-1, v.Get(1, 0))

					//                 Scale V(1,1) (= X(J1)) and/or V(2,1) (=X(J2))
					//                 to avoid overflow in updating right-hand side.
					xj = maxf64(math.Abs(v.Get(0, 0)), math.Abs(v.Get(1, 0)))
					if xj > one {
						rec = one / xj
						if maxf64(work.Get(j1-1), work.Get(j2-1)) > (bignum-xmax)*rec {
							goblas.Dscal(n, &rec, x, func() *int { y := 1; return &y }())
							(*scale) = (*scale) * rec
						}
					}

					//                 Update right-hand side
					if j1 > 1 {
						goblas.Daxpy(toPtr(j1-1), toPtrf64(-x.Get(j1-1)), t.Vector(0, j1-1), func() *int { y := 1; return &y }(), x, func() *int { y := 1; return &y }())
						goblas.Daxpy(toPtr(j1-1), toPtrf64(-x.Get(j2-1)), t.Vector(0, j2-1), func() *int { y := 1; return &y }(), x, func() *int { y := 1; return &y }())
						k = goblas.Idamax(toPtr(j1-1), x, func() *int { y := 1; return &y }())
						xmax = math.Abs(x.Get(k - 1))
					}

				}

			label30:
			}

		} else {
			//           Solve T**T*p = scale*c
			jnext = 1
			for j = 1; j <= (*n); j++ {
				if j < jnext {
					goto label40
				}
				j1 = j
				j2 = j
				jnext = j + 1
				if j < (*n) {
					if t.Get(j+1-1, j-1) != zero {
						j2 = j + 1
						jnext = j + 2
					}
				}

				if j1 == j2 {
					//                 1 by 1 diagonal block
					//
					//                 Scale if necessary to avoid overflow in forming the
					//                 right-hand side element by inner product.
					xj = math.Abs(x.Get(j1 - 1))
					if xmax > one {
						rec = one / xmax
						if work.Get(j1-1) > (bignum-xj)*rec {
							goblas.Dscal(n, &rec, x, func() *int { y := 1; return &y }())
							(*scale) = (*scale) * rec
							xmax = xmax * rec
						}
					}

					x.Set(j1-1, x.Get(j1-1)-goblas.Ddot(toPtr(j1-1), t.Vector(0, j1-1), func() *int { y := 1; return &y }(), x, func() *int { y := 1; return &y }()))

					xj = math.Abs(x.Get(j1 - 1))
					tjj = math.Abs(t.Get(j1-1, j1-1))
					tmp = t.Get(j1-1, j1-1)
					if tjj < smin {
						tmp = smin
						tjj = smin
						(*info) = 1
					}

					if tjj < one {
						if xj > bignum*tjj {
							rec = one / xj
							goblas.Dscal(n, &rec, x, func() *int { y := 1; return &y }())
							(*scale) = (*scale) * rec
							xmax = xmax * rec
						}
					}
					x.Set(j1-1, x.Get(j1-1)/tmp)
					xmax = maxf64(xmax, math.Abs(x.Get(j1-1)))

				} else {
					//                 2 by 2 diagonal block
					//
					//                 Scale if necessary to avoid overflow in forming the
					//                 right-hand side elements by inner product.
					xj = maxf64(math.Abs(x.Get(j1-1)), math.Abs(x.Get(j2-1)))
					if xmax > one {
						rec = one / xmax
						if maxf64(work.Get(j2-1), work.Get(j1-1)) > (bignum-xj)*rec {
							goblas.Dscal(n, &rec, x, func() *int { y := 1; return &y }())
							(*scale) = (*scale) * rec
							xmax = xmax * rec
						}
					}

					d.Set(0, 0, x.Get(j1-1)-goblas.Ddot(toPtr(j1-1), t.Vector(0, j1-1), func() *int { y := 1; return &y }(), x, func() *int { y := 1; return &y }()))
					d.Set(1, 0, x.Get(j2-1)-goblas.Ddot(toPtr(j1-1), t.Vector(0, j2-1), func() *int { y := 1; return &y }(), x, func() *int { y := 1; return &y }()))

					Dlaln2(true, func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), &smin, &one, t.Off(j1-1, j1-1), ldt, &one, &one, d, func() *int { y := 2; return &y }(), &zero, &zero, v, func() *int { y := 2; return &y }(), &scaloc, &xnorm, &ierr)
					if ierr != 0 {
						(*info) = 2
					}

					if scaloc != one {
						goblas.Dscal(n, &scaloc, x, func() *int { y := 1; return &y }())
						(*scale) = (*scale) * scaloc
					}
					x.Set(j1-1, v.Get(0, 0))
					x.Set(j2-1, v.Get(1, 0))
					xmax = maxf64(math.Abs(x.Get(j1-1)), math.Abs(x.Get(j2-1)), xmax)

				}
			label40:
			}
		}

	} else {

		sminw = maxf64(eps*math.Abs(*w), smin)
		if notran {
			//           Solve (T + iB)*(p+iq) = c+id
			jnext = (*n)
			for j = (*n); j >= 1; j-- {
				if j > jnext {
					goto label70
				}
				j1 = j
				j2 = j
				jnext = j - 1
				if j > 1 {
					if t.Get(j-1, j-1-1) != zero {
						j1 = j - 1
						jnext = j - 2
					}
				}

				if j1 == j2 {
					//                 1 by 1 diagonal block
					//
					//                 Scale if necessary to avoid overflow in division
					z = (*w)
					if j1 == 1 {
						z = b.Get(0)
					}
					xj = math.Abs(x.Get(j1-1)) + math.Abs(x.Get((*n)+j1-1))
					tjj = math.Abs(t.Get(j1-1, j1-1)) + math.Abs(z)
					tmp = t.Get(j1-1, j1-1)
					if tjj < sminw {
						tmp = sminw
						tjj = sminw
						(*info) = 1
					}

					if xj == zero {
						goto label70
					}

					if tjj < one {
						if xj > bignum*tjj {
							rec = one / xj
							goblas.Dscal(&n2, &rec, x, func() *int { y := 1; return &y }())
							(*scale) = (*scale) * rec
							xmax = xmax * rec
						}
					}
					Dladiv(x.GetPtr(j1-1), x.GetPtr((*n)+j1-1), &tmp, &z, &sr, &si)
					x.Set(j1-1, sr)
					x.Set((*n)+j1-1, si)
					xj = math.Abs(x.Get(j1-1)) + math.Abs(x.Get((*n)+j1-1))

					//                 Scale x if necessary to avoid overflow when adding a
					//                 multiple of column j1 of T.
					if xj > one {
						rec = one / xj
						if work.Get(j1-1) > (bignum-xmax)*rec {
							goblas.Dscal(&n2, &rec, x, func() *int { y := 1; return &y }())
							(*scale) = (*scale) * rec
						}
					}

					if j1 > 1 {
						goblas.Daxpy(toPtr(j1-1), toPtrf64(-x.Get(j1-1)), t.Vector(0, j1-1), func() *int { y := 1; return &y }(), x, func() *int { y := 1; return &y }())
						goblas.Daxpy(toPtr(j1-1), toPtrf64(-x.Get((*n)+j1-1)), t.Vector(0, j1-1), func() *int { y := 1; return &y }(), x.Off((*n)+1-1), func() *int { y := 1; return &y }())

						x.Set(0, x.Get(0)+b.Get(j1-1)*x.Get((*n)+j1-1))
						x.Set((*n)+1-1, x.Get((*n)+1-1)-b.Get(j1-1)*x.Get(j1-1))

						xmax = zero
						for k = 1; k <= j1-1; k++ {
							xmax = maxf64(xmax, math.Abs(x.Get(k-1))+math.Abs(x.Get(k+(*n)-1)))
						}
					}

				} else {
					//                 Meet 2 by 2 diagonal block
					d.Set(0, 0, x.Get(j1-1))
					d.Set(1, 0, x.Get(j2-1))
					d.Set(0, 1, x.Get((*n)+j1-1))
					d.Set(1, 1, x.Get((*n)+j2-1))
					Dlaln2(false, func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), &sminw, &one, t.Off(j1-1, j1-1), ldt, &one, &one, d, func() *int { y := 2; return &y }(), &zero, toPtrf64(-(*w)), v, func() *int { y := 2; return &y }(), &scaloc, &xnorm, &ierr)
					if ierr != 0 {
						(*info) = 2
					}

					if scaloc != one {
						goblas.Dscal(toPtr(2*(*n)), &scaloc, x, func() *int { y := 1; return &y }())
						(*scale) = scaloc * (*scale)
					}
					x.Set(j1-1, v.Get(0, 0))
					x.Set(j2-1, v.Get(1, 0))
					x.Set((*n)+j1-1, v.Get(0, 1))
					x.Set((*n)+j2-1, v.Get(1, 1))

					//                 Scale X(J1), .... to avoid overflow in
					//                 updating right hand side.
					xj = maxf64(math.Abs(v.Get(0, 0))+math.Abs(v.Get(0, 1)), math.Abs(v.Get(1, 0))+math.Abs(v.Get(1, 1)))
					if xj > one {
						rec = one / xj
						if maxf64(work.Get(j1-1), work.Get(j2-1)) > (bignum-xmax)*rec {
							goblas.Dscal(&n2, &rec, x, func() *int { y := 1; return &y }())
							(*scale) = (*scale) * rec
						}
					}

					//                 Update the right-hand side.
					if j1 > 1 {
						goblas.Daxpy(toPtr(j1-1), toPtrf64(-x.Get(j1-1)), t.Vector(0, j1-1), func() *int { y := 1; return &y }(), x, func() *int { y := 1; return &y }())
						goblas.Daxpy(toPtr(j1-1), toPtrf64(-x.Get(j2-1)), t.Vector(0, j2-1), func() *int { y := 1; return &y }(), x, func() *int { y := 1; return &y }())

						goblas.Daxpy(toPtr(j1-1), toPtrf64(-x.Get((*n)+j1-1)), t.Vector(0, j1-1), func() *int { y := 1; return &y }(), x.Off((*n)+1-1), func() *int { y := 1; return &y }())
						goblas.Daxpy(toPtr(j1-1), toPtrf64(-x.Get((*n)+j2-1)), t.Vector(0, j2-1), func() *int { y := 1; return &y }(), x.Off((*n)+1-1), func() *int { y := 1; return &y }())

						x.Set(0, x.Get(0)+b.Get(j1-1)*x.Get((*n)+j1-1)+b.Get(j2-1)*x.Get((*n)+j2-1))
						x.Set((*n)+1-1, x.Get((*n)+1-1)-b.Get(j1-1)*x.Get(j1-1)-b.Get(j2-1)*x.Get(j2-1))

						xmax = zero
						for k = 1; k <= j1-1; k++ {
							xmax = maxf64(math.Abs(x.Get(k-1))+math.Abs(x.Get(k+(*n)-1)), xmax)
						}
					}

				}
			label70:
			}

		} else {
			//           Solve (T + iB)**T*(p+iq) = c+id
			jnext = 1
			for j = 1; j <= (*n); j++ {
				if j < jnext {
					goto label80
				}
				j1 = j
				j2 = j
				jnext = j + 1
				if j < (*n) {
					if t.Get(j+1-1, j-1) != zero {
						j2 = j + 1
						jnext = j + 2
					}
				}

				if j1 == j2 {
					//                 1 by 1 diagonal block
					//
					//                 Scale if necessary to avoid overflow in forming the
					//                 right-hand side element by inner product.
					xj = math.Abs(x.Get(j1-1)) + math.Abs(x.Get(j1+(*n)-1))
					if xmax > one {
						rec = one / xmax
						if work.Get(j1-1) > (bignum-xj)*rec {
							goblas.Dscal(&n2, &rec, x, func() *int { y := 1; return &y }())
							(*scale) = (*scale) * rec
							xmax = xmax * rec
						}
					}

					x.Set(j1-1, x.Get(j1-1)-goblas.Ddot(toPtr(j1-1), t.Vector(0, j1-1), func() *int { y := 1; return &y }(), x, func() *int { y := 1; return &y }()))
					x.Set((*n)+j1-1, x.Get((*n)+j1-1)-goblas.Ddot(toPtr(j1-1), t.Vector(0, j1-1), func() *int { y := 1; return &y }(), x.Off((*n)+1-1), func() *int { y := 1; return &y }()))
					if j1 > 1 {
						x.Set(j1-1, x.Get(j1-1)-b.Get(j1-1)*x.Get((*n)+1-1))
						x.Set((*n)+j1-1, x.Get((*n)+j1-1)+b.Get(j1-1)*x.Get(0))
					}
					xj = math.Abs(x.Get(j1-1)) + math.Abs(x.Get(j1+(*n)-1))

					z = (*w)
					if j1 == 1 {
						z = b.Get(0)
					}

					//                 Scale if necessary to avoid overflow in
					//                 complex division
					tjj = math.Abs(t.Get(j1-1, j1-1)) + math.Abs(z)
					tmp = t.Get(j1-1, j1-1)
					if tjj < sminw {
						tmp = sminw
						tjj = sminw
						(*info) = 1
					}

					if tjj < one {
						if xj > bignum*tjj {
							rec = one / xj
							goblas.Dscal(&n2, &rec, x, func() *int { y := 1; return &y }())
							(*scale) = (*scale) * rec
							xmax = xmax * rec
						}
					}
					Dladiv(x.GetPtr(j1-1), x.GetPtr((*n)+j1-1), &tmp, toPtrf64(-z), &sr, &si)
					x.Set(j1-1, sr)
					x.Set(j1+(*n)-1, si)
					xmax = maxf64(math.Abs(x.Get(j1-1))+math.Abs(x.Get(j1+(*n)-1)), xmax)

				} else {
					//                 2 by 2 diagonal block
					//
					//                 Scale if necessary to avoid overflow in forming the
					//                 right-hand side element by inner product.
					xj = maxf64(math.Abs(x.Get(j1-1))+math.Abs(x.Get((*n)+j1-1)), math.Abs(x.Get(j2-1))+math.Abs(x.Get((*n)+j2-1)))
					if xmax > one {
						rec = one / xmax
						if maxf64(work.Get(j1-1), work.Get(j2-1)) > (bignum-xj)/xmax {
							goblas.Dscal(&n2, &rec, x, func() *int { y := 1; return &y }())
							(*scale) = (*scale) * rec
							xmax = xmax * rec
						}
					}

					d.Set(0, 0, x.Get(j1-1)-goblas.Ddot(toPtr(j1-1), t.Vector(0, j1-1), func() *int { y := 1; return &y }(), x, func() *int { y := 1; return &y }()))
					d.Set(1, 0, x.Get(j2-1)-goblas.Ddot(toPtr(j1-1), t.Vector(0, j2-1), func() *int { y := 1; return &y }(), x, func() *int { y := 1; return &y }()))
					d.Set(0, 1, x.Get((*n)+j1-1)-goblas.Ddot(toPtr(j1-1), t.Vector(0, j1-1), func() *int { y := 1; return &y }(), x.Off((*n)+1-1), func() *int { y := 1; return &y }()))
					d.Set(1, 1, x.Get((*n)+j2-1)-goblas.Ddot(toPtr(j1-1), t.Vector(0, j2-1), func() *int { y := 1; return &y }(), x.Off((*n)+1-1), func() *int { y := 1; return &y }()))
					d.Set(0, 0, d.Get(0, 0)-b.Get(j1-1)*x.Get((*n)+1-1))
					d.Set(1, 0, d.Get(1, 0)-b.Get(j2-1)*x.Get((*n)+1-1))
					d.Set(0, 1, d.Get(0, 1)+b.Get(j1-1)*x.Get(0))
					d.Set(1, 1, d.Get(1, 1)+b.Get(j2-1)*x.Get(0))

					Dlaln2(true, func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), &sminw, &one, t.Off(j1-1, j1-1), ldt, &one, &one, d, func() *int { y := 2; return &y }(), &zero, w, v, func() *int { y := 2; return &y }(), &scaloc, &xnorm, &ierr)
					if ierr != 0 {
						(*info) = 2
					}

					if scaloc != one {
						goblas.Dscal(&n2, &scaloc, x, func() *int { y := 1; return &y }())
						(*scale) = scaloc * (*scale)
					}
					x.Set(j1-1, v.Get(0, 0))
					x.Set(j2-1, v.Get(1, 0))
					x.Set((*n)+j1-1, v.Get(0, 1))
					x.Set((*n)+j2-1, v.Get(1, 1))
					xmax = maxf64(math.Abs(x.Get(j1-1))+math.Abs(x.Get((*n)+j1-1)), math.Abs(x.Get(j2-1))+math.Abs(x.Get((*n)+j2-1)), xmax)

				}

			label80:
			}

		}

	}
}
