package golapack

import (
	"math/cmplx"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zhbtrd reduces a complex Hermitian band matrix A to real symmetric
// tridiagonal form T by a unitary similarity transformation:
// Q**H * A * Q = T.
func Zhbtrd(vect, uplo byte, n, kd *int, ab *mat.CMatrix, ldab *int, d, e *mat.Vector, q *mat.CMatrix, ldq *int, work *mat.CVector, info *int) {
	var initq, upper, wantq bool
	var cone, czero, t, temp complex128
	var abst, zero float64
	var i, i2, ibl, inca, incx, iqaend, iqb, iqend, j, j1, j1end, j1inc, j2, jend, jin, jinc, k, kd1, kdm1, kdn, l, last, lend, nq, nr, nrt int

	zero = 0.0
	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)

	//     Test the input parameters
	initq = vect == 'V'
	wantq = initq || vect == 'U'
	upper = uplo == 'U'
	kd1 = (*kd) + 1
	kdm1 = (*kd) - 1
	incx = (*ldab) - 1
	iqend = 1

	(*info) = 0
	if !wantq && vect != 'N' {
		(*info) = -1
	} else if !upper && uplo != 'L' {
		(*info) = -2
	} else if (*n) < 0 {
		(*info) = -3
	} else if (*kd) < 0 {
		(*info) = -4
	} else if (*ldab) < kd1 {
		(*info) = -6
	} else if (*ldq) < max(1, *n) && wantq {
		(*info) = -10
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZHBTRD"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	//     Initialize Q to the unit matrix, if needed
	if initq {
		Zlaset('F', n, n, &czero, &cone, q, ldq)
	}

	//     Wherever possible, plane rotations are generated and applied in
	//     vector operations of length NR over the index set J1:J2:KD1.
	//
	//     The real cosines and complex sines of the plane rotations are
	//     stored in the arrays D and WORK.
	inca = kd1 * (*ldab)
	kdn = min((*n)-1, *kd)
	if upper {

		if (*kd) > 1 {
			//           Reduce to complex Hermitian tridiagonal form, working with
			//           the upper triangle
			nr = 0
			j1 = kdn + 2
			j2 = 1

			ab.Set(kd1-1, 0, ab.GetReCmplx(kd1-1, 0))
			for i = 1; i <= (*n)-2; i++ {
				//              Reduce i-th row of matrix to tridiagonal form
				for k = kdn + 1; k >= 2; k-- {
					j1 = j1 + kdn
					j2 = j2 + kdn

					if nr > 0 {
						//                    generate plane rotations to annihilate nonzero
						//                    elements which have been created outside the band
						Zlargv(&nr, ab.CVector(0, j1-1-1), &inca, work.Off(j1-1), &kd1, d.Off(j1-1), &kd1)

						//                    apply rotations from the right
						//
						//
						//                    Dependent on the the number of diagonals either
						//                    ZLARTV or ZROT is used
						if nr >= 2*(*kd)-1 {
							for l = 1; l <= (*kd)-1; l++ {
								Zlartv(&nr, ab.CVector(l, j1-1-1), &inca, ab.CVector(l-1, j1-1), &inca, d.Off(j1-1), work.Off(j1-1), &kd1)
							}

						} else {
							jend = j1 + (nr-1)*kd1
							for jinc = j1; jinc <= jend; jinc += kd1 {
								Zrot(&kdm1, ab.CVector(1, jinc-1-1), func() *int { y := 1; return &y }(), ab.CVector(0, jinc-1), func() *int { y := 1; return &y }(), d.GetPtr(jinc-1), work.GetPtr(jinc-1))
							}
						}
					}

					if k > 2 {
						if k <= (*n)-i+1 {
							//                       generate plane rotation to annihilate a(i,i+k-1)
							//                       within the band
							Zlartg(ab.GetPtr((*kd)-k+3-1, i+k-2-1), ab.GetPtr((*kd)-k+2-1, i+k-1-1), d.GetPtr(i+k-1-1), work.GetPtr(i+k-1-1), &temp)
							ab.Set((*kd)-k+3-1, i+k-2-1, temp)

							//                       apply rotation from the right
							Zrot(toPtr(k-3), ab.CVector((*kd)-k+4-1, i+k-2-1), func() *int { y := 1; return &y }(), ab.CVector((*kd)-k+3-1, i+k-1-1), func() *int { y := 1; return &y }(), d.GetPtr(i+k-1-1), work.GetPtr(i+k-1-1))
						}
						nr = nr + 1
						j1 = j1 - kdn - 1
					}

					//                 apply plane rotations from both sides to diagonal
					//                 blocks
					if nr > 0 {
						Zlar2v(&nr, ab.CVector(kd1-1, j1-1-1), ab.CVector(kd1-1, j1-1), ab.CVector((*kd)-1, j1-1), &inca, d.Off(j1-1), work.Off(j1-1), &kd1)
					}

					//                 apply plane rotations from the left
					if nr > 0 {
						Zlacgv(&nr, work.Off(j1-1), &kd1)
						if 2*(*kd)-1 < nr {
							//                    Dependent on the the number of diagonals either
							//                    ZLARTV or ZROT is used
							for l = 1; l <= (*kd)-1; l++ {
								if j2+l > (*n) {
									nrt = nr - 1
								} else {
									nrt = nr
								}
								if nrt > 0 {
									Zlartv(&nrt, ab.CVector((*kd)-l-1, j1+l-1), &inca, ab.CVector((*kd)-l, j1+l-1), &inca, d.Off(j1-1), work.Off(j1-1), &kd1)
								}
							}
						} else {
							j1end = j1 + kd1*(nr-2)
							if j1end >= j1 {
								for jin = j1; jin <= j1end; jin += kd1 {
									Zrot(toPtr((*kd)-1), ab.CVector((*kd)-1-1, jin), &incx, ab.CVector((*kd)-1, jin), &incx, d.GetPtr(jin-1), work.GetPtr(jin-1))
								}
							}
							lend = min(kdm1, (*n)-j2)
							last = j1end + kd1
							if lend > 0 {
								Zrot(&lend, ab.CVector((*kd)-1-1, last), &incx, ab.CVector((*kd)-1, last), &incx, d.GetPtr(last-1), work.GetPtr(last-1))
							}
						}
					}

					if wantq {
						//                    accumulate product of plane rotations in Q
						if initq {
							//                 take advantage of the fact that Q was
							//                 initially the Identity matrix
							iqend = max(iqend, j2)
							i2 = max(0, k-3)
							iqaend = 1 + i*(*kd)
							if k == 2 {
								iqaend = iqaend + (*kd)
							}
							iqaend = min(iqaend, iqend)
							for j = j1; j <= j2; j += kd1 {
								ibl = i - i2/kdm1
								i2 = i2 + 1
								iqb = max(1, j-ibl)
								nq = 1 + iqaend - iqb
								iqaend = min(iqaend+(*kd), iqend)
								Zrot(&nq, q.CVector(iqb-1, j-1-1), func() *int { y := 1; return &y }(), q.CVector(iqb-1, j-1), func() *int { y := 1; return &y }(), d.GetPtr(j-1), toPtrc128(work.GetConj(j-1)))
							}
						} else {

							for j = j1; j <= j2; j += kd1 {
								Zrot(n, q.CVector(0, j-1-1), func() *int { y := 1; return &y }(), q.CVector(0, j-1), func() *int { y := 1; return &y }(), d.GetPtr(j-1), toPtrc128(work.GetConj(j-1)))
							}
						}

					}

					if j2+kdn > (*n) {
						//                    adjust J2 to keep within the bounds of the matrix
						nr = nr - 1
						j2 = j2 - kdn - 1
					}

					for j = j1; j <= j2; j += kd1 {
						//                    create nonzero element a(j-1,j+kd) outside the band
						//                    and store it in WORK
						work.Set(j+(*kd)-1, work.Get(j-1)*ab.Get(0, j+(*kd)-1))
						ab.Set(0, j+(*kd)-1, d.GetCmplx(j-1)*ab.Get(0, j+(*kd)-1))
					}
				}
			}
		}

		if (*kd) > 0 {
			//           make off-diagonal elements real and copy them to E
			for i = 1; i <= (*n)-1; i++ {
				t = ab.Get((*kd)-1, i)
				abst = cmplx.Abs(t)
				ab.SetRe((*kd)-1, i, abst)
				e.Set(i-1, abst)
				if abst != zero {
					t = t / complex(abst, 0)
				} else {
					t = cone
				}
				if i < (*n)-1 {
					ab.Set((*kd)-1, i+2-1, ab.Get((*kd)-1, i+2-1)*t)
				}
				if wantq {
					goblas.Zscal(*n, cmplx.Conj(t), q.CVector(0, i, 1))
				}
			}
		} else {
			//           set E to zero if original matrix was diagonal
			for i = 1; i <= (*n)-1; i++ {
				e.Set(i-1, zero)
			}
		}

		//        copy diagonal elements to D
		for i = 1; i <= (*n); i++ {
			d.Set(i-1, ab.GetRe(kd1-1, i-1))
		}

	} else {

		if (*kd) > 1 {
			//           Reduce to complex Hermitian tridiagonal form, working with
			//           the lower triangle
			nr = 0
			j1 = kdn + 2
			j2 = 1

			ab.Set(0, 0, ab.GetReCmplx(0, 0))
			for i = 1; i <= (*n)-2; i++ {
				//              Reduce i-th column of matrix to tridiagonal form
				for k = kdn + 1; k >= 2; k-- {
					j1 = j1 + kdn
					j2 = j2 + kdn

					if nr > 0 {
						//                    generate plane rotations to annihilate nonzero
						//                    elements which have been created outside the band
						Zlargv(&nr, ab.CVector(kd1-1, j1-kd1-1), &inca, work.Off(j1-1), &kd1, d.Off(j1-1), &kd1)

						//                    apply plane rotations from one side
						//
						//
						//                    Dependent on the the number of diagonals either
						//                    ZLARTV or ZROT is used
						if nr > 2*(*kd)-1 {
							for l = 1; l <= (*kd)-1; l++ {
								Zlartv(&nr, ab.CVector(kd1-l-1, j1-kd1+l-1), &inca, ab.CVector(kd1-l, j1-kd1+l-1), &inca, d.Off(j1-1), work.Off(j1-1), &kd1)
							}
						} else {
							jend = j1 + kd1*(nr-1)
							for jinc = j1; jinc <= jend; jinc += kd1 {
								Zrot(&kdm1, ab.CVector((*kd)-1, jinc-(*kd)-1), &incx, ab.CVector(kd1-1, jinc-(*kd)-1), &incx, d.GetPtr(jinc-1), work.GetPtr(jinc-1))
							}
						}

					}

					if k > 2 {
						if k <= (*n)-i+1 {
							//                       generate plane rotation to annihilate a(i+k-1,i)
							//                       within the band
							Zlartg(ab.GetPtr(k-1-1, i-1), ab.GetPtr(k-1, i-1), d.GetPtr(i+k-1-1), work.GetPtr(i+k-1-1), &temp)
							ab.Set(k-1-1, i-1, temp)

							//                       apply rotation from the left
							Zrot(toPtr(k-3), ab.CVector(k-2-1, i), toPtr((*ldab)-1), ab.CVector(k-1-1, i), toPtr((*ldab)-1), d.GetPtr(i+k-1-1), work.GetPtr(i+k-1-1))
						}
						nr = nr + 1
						j1 = j1 - kdn - 1
					}

					//                 apply plane rotations from both sides to diagonal
					//                 blocks
					if nr > 0 {
						Zlar2v(&nr, ab.CVector(0, j1-1-1), ab.CVector(0, j1-1), ab.CVector(1, j1-1-1), &inca, d.Off(j1-1), work.Off(j1-1), &kd1)
					}

					//                 apply plane rotations from the right
					//
					//
					//                    Dependent on the the number of diagonals either
					//                    ZLARTV or ZROT is used
					if nr > 0 {
						Zlacgv(&nr, work.Off(j1-1), &kd1)
						if nr > 2*(*kd)-1 {
							for l = 1; l <= (*kd)-1; l++ {
								if j2+l > (*n) {
									nrt = nr - 1
								} else {
									nrt = nr
								}
								if nrt > 0 {
									Zlartv(&nrt, ab.CVector(l+2-1, j1-1-1), &inca, ab.CVector(l, j1-1), &inca, d.Off(j1-1), work.Off(j1-1), &kd1)
								}
							}
						} else {
							j1end = j1 + kd1*(nr-2)
							if j1end >= j1 {
								for j1inc = j1; j1inc <= j1end; j1inc += kd1 {
									Zrot(&kdm1, ab.CVector(2, j1inc-1-1), func() *int { y := 1; return &y }(), ab.CVector(1, j1inc-1), func() *int { y := 1; return &y }(), d.GetPtr(j1inc-1), work.GetPtr(j1inc-1))
								}
							}
							lend = min(kdm1, (*n)-j2)
							last = j1end + kd1
							if lend > 0 {
								Zrot(&lend, ab.CVector(2, last-1-1), func() *int { y := 1; return &y }(), ab.CVector(1, last-1), func() *int { y := 1; return &y }(), d.GetPtr(last-1), work.GetPtr(last-1))
							}
						}
					}

					//
					if wantq {
						//                    accumulate product of plane rotations in Q
						if initq {
							//                 take advantage of the fact that Q was
							//                 initially the Identity matrix
							iqend = max(iqend, j2)
							i2 = max(0, k-3)
							iqaend = 1 + i*(*kd)
							if k == 2 {
								iqaend = iqaend + (*kd)
							}
							iqaend = min(iqaend, iqend)
							for j = j1; j <= j2; j += kd1 {
								ibl = i - i2/kdm1
								i2 = i2 + 1
								iqb = max(1, j-ibl)
								nq = 1 + iqaend - iqb
								iqaend = min(iqaend+(*kd), iqend)
								Zrot(&nq, q.CVector(iqb-1, j-1-1), func() *int { y := 1; return &y }(), q.CVector(iqb-1, j-1), func() *int { y := 1; return &y }(), d.GetPtr(j-1), work.GetPtr(j-1))
							}
						} else {

							for j = j1; j <= j2; j += kd1 {
								Zrot(n, q.CVector(0, j-1-1), func() *int { y := 1; return &y }(), q.CVector(0, j-1), func() *int { y := 1; return &y }(), d.GetPtr(j-1), work.GetPtr(j-1))
							}
						}
					}

					if j2+kdn > (*n) {
						//                    adjust J2 to keep within the bounds of the matrix
						nr = nr - 1
						j2 = j2 - kdn - 1
					}

					for j = j1; j <= j2; j += kd1 {
						//                    create nonzero element a(j+kd,j-1) outside the
						//                    band and store it in WORK
						work.Set(j+(*kd)-1, work.Get(j-1)*ab.Get(kd1-1, j-1))
						ab.Set(kd1-1, j-1, d.GetCmplx(j-1)*ab.Get(kd1-1, j-1))
					}
				}
			}
		}

		if (*kd) > 0 {
			//           make off-diagonal elements real and copy them to E
			for i = 1; i <= (*n)-1; i++ {
				t = ab.Get(1, i-1)
				abst = cmplx.Abs(t)
				ab.SetRe(1, i-1, abst)
				e.Set(i-1, abst)
				if abst != zero {
					t = t / complex(abst, 0)
				} else {
					t = cone
				}
				if i < (*n)-1 {
					ab.Set(1, i, ab.Get(1, i)*t)
				}
				if wantq {
					goblas.Zscal(*n, t, q.CVector(0, i, 1))
				}
			}
		} else {
			//           set E to zero if original matrix was diagonal
			for i = 1; i <= (*n)-1; i++ {
				e.Set(i-1, zero)
			}
		}

		//        copy diagonal elements to D
		for i = 1; i <= (*n); i++ {
			d.Set(i-1, ab.GetRe(0, i-1))
		}
	}
}
