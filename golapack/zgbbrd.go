package golapack

import (
	"math/cmplx"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgbbrd reduces a complex general m-by-n band matrix A to real upper
// bidiagonal form B by a unitary transformation: Q**H * A * P = B.
//
// The routine computes B, and optionally forms Q or P**H, or computes
// Q**H*C for a given matrix C.
func Zgbbrd(vect byte, m, n, ncc, kl, ku *int, ab *mat.CMatrix, ldab *int, d, e *mat.Vector, q *mat.CMatrix, ldq *int, pt *mat.CMatrix, ldpt *int, c *mat.CMatrix, ldc *int, work *mat.CVector, rwork *mat.Vector, info *int) {
	var wantb, wantc, wantpt, wantq bool
	var cone, czero, ra, rb, rs, t complex128
	var abst, rc, zero float64
	var i, inca, j, j1, j2, kb, kb1, kk, klm, klu1, kun, l, minmn, ml, ml0, mu, mu0, nr, nrt int

	zero = 0.0
	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)

	//     Test the input parameters
	wantb = vect == 'B'
	wantq = vect == 'Q' || wantb
	wantpt = vect == 'P' || wantb
	wantc = (*ncc) > 0
	klu1 = (*kl) + (*ku) + 1
	(*info) = 0
	if !wantq && !wantpt && vect != 'N' {
		(*info) = -1
	} else if (*m) < 0 {
		(*info) = -2
	} else if (*n) < 0 {
		(*info) = -3
	} else if (*ncc) < 0 {
		(*info) = -4
	} else if (*kl) < 0 {
		(*info) = -5
	} else if (*ku) < 0 {
		(*info) = -6
	} else if (*ldab) < klu1 {
		(*info) = -8
	} else if (*ldq) < 1 || wantq && (*ldq) < maxint(1, *m) {
		(*info) = -12
	} else if (*ldpt) < 1 || wantpt && (*ldpt) < maxint(1, *n) {
		(*info) = -14
	} else if (*ldc) < 1 || wantc && (*ldc) < maxint(1, *m) {
		(*info) = -16
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZGBBRD"), -(*info))
		return
	}

	//     Initialize Q and P**H to the unit matrix, if needed
	if wantq {
		Zlaset('F', m, m, &czero, &cone, q, ldq)
	}
	if wantpt {
		Zlaset('F', n, n, &czero, &cone, pt, ldpt)
	}

	//     Quick return if possible.
	if (*m) == 0 || (*n) == 0 {
		return
	}

	minmn = minint(*m, *n)

	if (*kl)+(*ku) > 1 {
		//        Reduce to upper bidiagonal form if KU > 0; if KU = 0, reduce
		//        first to lower bidiagonal form and then transform to upper
		//        bidiagonal
		if (*ku) > 0 {
			ml0 = 1
			mu0 = 2
		} else {
			ml0 = 2
			mu0 = 1
		}

		//        Wherever possible, plane rotations are generated and applied in
		//        vector operations of length NR over the index set J1:J2:KLU1.
		//
		//        The complex sines of the plane rotations are stored in WORK,
		//        and the real cosines in RWORK.
		klm = minint((*m)-1, *kl)
		kun = minint((*n)-1, *ku)
		kb = klm + kun
		kb1 = kb + 1
		inca = kb1 * (*ldab)
		nr = 0
		j1 = klm + 2
		j2 = 1 - kun

		for i = 1; i <= minmn; i++ {
			//           Reduce i-th column and i-th row of matrix to bidiagonal form
			ml = klm + 1
			mu = kun + 1
			for kk = 1; kk <= kb; kk++ {
				j1 = j1 + kb
				j2 = j2 + kb

				//              generate plane rotations to annihilate nonzero elements
				//              which have been created below the band
				if nr > 0 {
					Zlargv(&nr, ab.CVector(klu1-1, j1-klm-1-1), &inca, work.Off(j1-1), &kb1, rwork.Off(j1-1), &kb1)
				}

				//              apply plane rotations from the left
				for l = 1; l <= kb; l++ {
					if j2-klm+l-1 > (*n) {
						nrt = nr - 1
					} else {
						nrt = nr
					}
					if nrt > 0 {
						Zlartv(&nrt, ab.CVector(klu1-l-1, j1-klm+l-1-1), &inca, ab.CVector(klu1-l+1-1, j1-klm+l-1-1), &inca, rwork.Off(j1-1), work.Off(j1-1), &kb1)
					}
				}

				if ml > ml0 {
					if ml <= (*m)-i+1 {
						//                    generate plane rotation to annihilate a(i+ml-1,i)
						//                    within the band, and apply rotation from the left
						Zlartg(ab.GetPtr((*ku)+ml-1-1, i-1), ab.GetPtr((*ku)+ml-1, i-1), rwork.GetPtr(i+ml-1-1), work.GetPtr(i+ml-1-1), &ra)
						ab.Set((*ku)+ml-1-1, i-1, ra)
						if i < (*n) {
							Zrot(toPtr(minint((*ku)+ml-2, (*n)-i)), ab.CVector((*ku)+ml-2-1, i+1-1), toPtr((*ldab)-1), ab.CVector((*ku)+ml-1-1, i+1-1), toPtr((*ldab)-1), rwork.GetPtr(i+ml-1-1), work.GetPtr(i+ml-1-1))
						}
					}
					nr = nr + 1
					j1 = j1 - kb1
				}

				if wantq {
					//                 accumulate product of plane rotations in Q
					for j = j1; j <= j2; j += kb1 {
						Zrot(m, q.CVector(0, j-1-1), func() *int { y := 1; return &y }(), q.CVector(0, j-1), func() *int { y := 1; return &y }(), rwork.GetPtr(j-1), toPtrc128(work.GetConj(j-1)))
					}
				}

				if wantc {
					//                 apply plane rotations to C
					for j = j1; j <= j2; j += kb1 {
						Zrot(ncc, c.CVector(j-1-1, 0), ldc, c.CVector(j-1, 0), ldc, rwork.GetPtr(j-1), work.GetPtr(j-1))
					}
				}

				if j2+kun > (*n) {
					//                 adjust J2 to keep within the bounds of the matrix
					nr = nr - 1
					j2 = j2 - kb1
				}

				for j = j1; j <= j2; j += kb1 {
					//                 create nonzero element a(j-1,j+ku) above the band
					//                 and store it in WORK(n+1:2*n)
					work.Set(j+kun-1, work.Get(j-1)*ab.Get(0, j+kun-1))
					ab.Set(0, j+kun-1, rwork.GetCmplx(j-1)*ab.Get(0, j+kun-1))
				}

				//              generate plane rotations to annihilate nonzero elements
				//              which have been generated above the band
				if nr > 0 {
					Zlargv(&nr, ab.CVector(0, j1+kun-1-1), &inca, work.Off(j1+kun-1), &kb1, rwork.Off(j1+kun-1), &kb1)
				}

				//              apply plane rotations from the right
				for l = 1; l <= kb; l++ {
					if j2+l-1 > (*m) {
						nrt = nr - 1
					} else {
						nrt = nr
					}
					if nrt > 0 {
						Zlartv(&nrt, ab.CVector(l+1-1, j1+kun-1-1), &inca, ab.CVector(l-1, j1+kun-1), &inca, rwork.Off(j1+kun-1), work.Off(j1+kun-1), &kb1)
					}
				}

				if ml == ml0 && mu > mu0 {
					if mu <= (*n)-i+1 {
						//                    generate plane rotation to annihilate a(i,i+mu-1)
						//                    within the band, and apply rotation from the right
						Zlartg(ab.GetPtr((*ku)-mu+3-1, i+mu-2-1), ab.GetPtr((*ku)-mu+2-1, i+mu-1-1), rwork.GetPtr(i+mu-1-1), work.GetPtr(i+mu-1-1), &ra)
						ab.Set((*ku)-mu+3-1, i+mu-2-1, ra)
						Zrot(toPtr(minint((*kl)+mu-2, (*m)-i)), ab.CVector((*ku)-mu+4-1, i+mu-2-1), func() *int { y := 1; return &y }(), ab.CVector((*ku)-mu+3-1, i+mu-1-1), func() *int { y := 1; return &y }(), rwork.GetPtr(i+mu-1-1), work.GetPtr(i+mu-1-1))
					}
					nr = nr + 1
					j1 = j1 - kb1
				}

				if wantpt {
					//                 accumulate product of plane rotations in P**H
					for j = j1; j <= j2; j += kb1 {
						Zrot(n, pt.CVector(j+kun-1-1, 0), ldpt, pt.CVector(j+kun-1, 0), ldpt, rwork.GetPtr(j+kun-1), toPtrc128(work.GetConj(j+kun-1)))
					}
				}

				if j2+kb > (*m) {
					//                 adjust J2 to keep within the bounds of the matrix
					nr = nr - 1
					j2 = j2 - kb1
				}

				for j = j1; j <= j2; j += kb1 {
					//                 create nonzero element a(j+kl+ku,j+ku-1) below the
					//                 band and store it in WORK(1:n)
					work.Set(j+kb-1, work.Get(j+kun-1)*ab.Get(klu1-1, j+kun-1))
					ab.Set(klu1-1, j+kun-1, rwork.GetCmplx(j+kun-1)*ab.Get(klu1-1, j+kun-1))
				}

				if ml > ml0 {
					ml = ml - 1
				} else {
					mu = mu - 1
				}
			}
		}
	}

	if (*ku) == 0 && (*kl) > 0 {
		//        A has been reduced to complex lower bidiagonal form
		//
		//        Transform lower bidiagonal form to upper bidiagonal by applying
		//        plane rotations from the left, overwriting superdiagonal
		//        elements on subdiagonal elements
		for i = 1; i <= minint((*m)-1, *n); i++ {
			Zlartg(ab.GetPtr(0, i-1), ab.GetPtr(1, i-1), &rc, &rs, &ra)
			ab.Set(0, i-1, ra)
			if i < (*n) {
				ab.Set(1, i-1, rs*ab.Get(0, i+1-1))
				ab.Set(0, i+1-1, toCmplx(rc)*ab.Get(0, i+1-1))
			}
			if wantq {
				Zrot(m, q.CVector(0, i-1), func() *int { y := 1; return &y }(), q.CVector(0, i+1-1), func() *int { y := 1; return &y }(), &rc, toPtrc128(cmplx.Conj(rs)))
			}
			if wantc {
				Zrot(ncc, c.CVector(i-1, 0), ldc, c.CVector(i+1-1, 0), ldc, &rc, &rs)
			}
		}
	} else {
		//        A has been reduced to complex upper bidiagonal form or is
		//        diagonal
		if (*ku) > 0 && (*m) < (*n) {
			//           Annihilate a(m,m+1) by applying plane rotations from the
			//           right
			rb = ab.Get((*ku)-1, (*m)+1-1)
			for i = (*m); i >= 1; i-- {
				Zlartg(ab.GetPtr((*ku)+1-1, i-1), &rb, &rc, &rs, &ra)
				ab.Set((*ku)+1-1, i-1, ra)
				if i > 1 {
					rb = -cmplx.Conj(rs) * ab.Get((*ku)-1, i-1)
					ab.Set((*ku)-1, i-1, toCmplx(rc)*ab.Get((*ku)-1, i-1))
				}
				if wantpt {
					Zrot(n, pt.CVector(i-1, 0), ldpt, pt.CVector((*m)+1-1, 0), ldpt, &rc, toPtrc128(cmplx.Conj(rs)))
				}
			}
		}
	}

	//     Make diagonal and superdiagonal elements real, storing them in D
	//     and E
	t = ab.Get((*ku)+1-1, 0)
	for i = 1; i <= minmn; i++ {
		abst = cmplx.Abs(t)
		d.Set(i-1, abst)
		if abst != zero {
			t = t / toCmplx(abst)
		} else {
			t = cone
		}
		if wantq {
			goblas.Zscal(*m, t, q.CVector(0, i-1), 1)
		}
		if wantc {
			goblas.Zscal(*ncc, cmplx.Conj(t), c.CVector(i-1, 0), *ldc)
		}
		if i < minmn {
			if (*ku) == 0 && (*kl) == 0 {
				e.Set(i-1, zero)
				t = ab.Get(0, i+1-1)
			} else {
				if (*ku) == 0 {
					t = ab.Get(1, i-1) * cmplx.Conj(t)
				} else {
					t = ab.Get((*ku)-1, i+1-1) * cmplx.Conj(t)
				}
				abst = cmplx.Abs(t)
				e.Set(i-1, abst)
				if abst != zero {
					t = t / toCmplx(abst)
				} else {
					t = cone
				}
				if wantpt {
					goblas.Zscal(*n, t, pt.CVector(i+1-1, 0), *ldpt)
				}
				t = ab.Get((*ku)+1-1, i+1-1) * cmplx.Conj(t)
			}
		}
	}
}
