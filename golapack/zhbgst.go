package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Zhbgst reduces a complex Hermitian-definite banded generalized
// eigenproblem  A*x = lambda*B*x  to standard form  C*y = lambda*y,
// such that C has the same bandwidth as A.
//
// B must have been previously factorized as S**H*S by ZPBSTF, using a
// split Cholesky factorization. A is overwritten by C = X**H*A*X, where
// X = S**(-1)*Q and Q is a unitary matrix chosen to preserve the
// bandwidth of A.
func Zhbgst(vect, uplo byte, n, ka, kb *int, ab *mat.CMatrix, ldab *int, bb *mat.CMatrix, ldbb *int, x *mat.CMatrix, ldx *int, work *mat.CVector, rwork *mat.Vector, info *int) {
	var update, upper, wantx bool
	var cone, czero, ra, ra1, t complex128
	var bii, one float64
	var i, i0, i1, i2, inca, j, j1, j1t, j2, j2t, k, ka1, kb1, kbt, l, m, nr, nrt, nx int

	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)
	one = 1.0

	//     Test the input parameters
	wantx = vect == 'V'
	upper = uplo == 'U'
	ka1 = (*ka) + 1
	kb1 = (*kb) + 1
	(*info) = 0
	if !wantx && vect != 'N' {
		(*info) = -1
	} else if !upper && uplo != 'L' {
		(*info) = -2
	} else if (*n) < 0 {
		(*info) = -3
	} else if (*ka) < 0 {
		(*info) = -4
	} else if (*kb) < 0 || (*kb) > (*ka) {
		(*info) = -5
	} else if (*ldab) < (*ka)+1 {
		(*info) = -7
	} else if (*ldbb) < (*kb)+1 {
		(*info) = -9
	} else if (*ldx) < 1 || wantx && (*ldx) < maxint(1, *n) {
		(*info) = -11
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZHBGST"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	inca = (*ldab) * ka1

	//     Initialize X to the unit matrix, if needed
	if wantx {
		Zlaset('F', n, n, &czero, &cone, x, ldx)
	}

	//     Set M to the splitting point m. It must be the same value as is
	//     used in ZPBSTF. The chosen value allows the arrays WORK and RWORK
	//     to be of dimension (N).
	m = ((*n) + (*kb)) / 2

	//     The routine works in two phases, corresponding to the two halves
	//     of the split Cholesky factorization of B as S**H*S where
	//
	//     S = ( U    )
	//         ( M  L )
	//
	//     with U upper triangular of order m, and L lower triangular of
	//     order n-m. S has the same bandwidth as B.
	//
	//     S is treated as a product of elementary matrices:
	//
	//     S = S(m)*S(m-1)*...*S(2)*S(1)*S(m+1)*S(m+2)*...*S(n-1)*S(n)
	//
	//     where S(i) is determined by the i-th row of S.
	//
	//     In phase 1, the index i takes the values n, n-1, ... , m+1;
	//     in phase 2, it takes the values 1, 2, ... , m.
	//
	//     For each value of i, the current matrix A is updated by forming
	//     inv(S(i))**H*A*inv(S(i)). This creates a triangular bulge outside
	//     the band of A. The bulge is then pushed down toward the bottom of
	//     A in phase 1, and up toward the top of A in phase 2, by applying
	//     plane rotations.
	//
	//     There are kb*(kb+1)/2 elements in the bulge, but at most 2*kb-1
	//     of them are linearly independent, so annihilating a bulge requires
	//     only 2*kb-1 plane rotations. The rotations are divided into a 1st
	//     set of kb-1 rotations, and a 2nd set of kb rotations.
	//
	//     Wherever possible, rotations are generated and applied in vector
	//     operations of length NR between the indices J1 and J2 (sometimes
	//     replaced by modified values NRT, J1T or J2T).
	//
	//     The real cosines and complex sines of the rotations are stored in
	//     the arrays RWORK and WORK, those of the 1st set in elements
	//     2:m-kb-1, and those of the 2nd set in elements m-kb+1:n.
	//
	//     The bulges are not formed explicitly; nonzero elements outside the
	//     band are created only when they are required for generating new
	//     rotations; they are stored in the array WORK, in positions where
	//     they are later overwritten by the sines of the rotations which
	//     annihilate them.
	//
	//     **************************** Phase 1 *****************************
	//
	//     The logical structure of this phase is:
	//
	//     UPDATE = .TRUE.
	//     DO I = N, M + 1, -1
	//        use S(i) to update A and create a new bulge
	//        apply rotations to push all bulges KA positions downward
	//     END DO
	//     UPDATE = .FALSE.
	//     DO I = M + KA + 1, N - 1
	//        apply rotations to push all bulges KA positions downward
	//     END DO
	//
	//     To avoid duplicating code, the two loops are merged.
	update = true
	i = (*n) + 1
label10:
	;
	if update {
		i = i - 1
		kbt = minint(*kb, i-1)
		i0 = i - 1
		i1 = minint(*n, i+(*ka))
		i2 = i - kbt + ka1
		if i < m+1 {
			update = false
			i = i + 1
			i0 = m
			if (*ka) == 0 {
				goto label480
			}
			goto label10
		}
	} else {
		i = i + (*ka)
		if i > (*n)-1 {
			goto label480
		}
	}

	if upper {
		//        Transform A, working with the upper triangle
		if update {
			//           Form  inv(S(i))**H * A * inv(S(i))
			bii = bb.GetRe(kb1-1, i-1)
			ab.SetRe(ka1-1, i-1, (ab.GetRe(ka1-1, i-1)/bii)/bii)
			for j = i + 1; j <= i1; j++ {
				ab.Set(i-j+ka1-1, j-1, ab.Get(i-j+ka1-1, j-1)/complex(bii, 0))
			}
			for j = maxint(1, i-(*ka)); j <= i-1; j++ {
				ab.Set(j-i+ka1-1, i-1, ab.Get(j-i+ka1-1, i-1)/complex(bii, 0))
			}
			for k = i - kbt; k <= i-1; k++ {
				for j = i - kbt; j <= k; j++ {
					ab.Set(j-k+ka1-1, k-1, ab.Get(j-k+ka1-1, k-1)-bb.Get(j-i+kb1-1, i-1)*ab.GetConj(k-i+ka1-1, i-1)-bb.GetConj(k-i+kb1-1, i-1)*ab.Get(j-i+ka1-1, i-1)+ab.GetReCmplx(ka1-1, i-1)*bb.Get(j-i+kb1-1, i-1)*bb.GetConj(k-i+kb1-1, i-1))
				}
				for j = maxint(1, i-(*ka)); j <= i-kbt-1; j++ {
					ab.Set(j-k+ka1-1, k-1, ab.Get(j-k+ka1-1, k-1)-bb.GetConj(k-i+kb1-1, i-1)*ab.Get(j-i+ka1-1, i-1))
				}
			}
			for j = i; j <= i1; j++ {
				for k = maxint(j-(*ka), i-kbt); k <= i-1; k++ {
					ab.Set(k-j+ka1-1, j-1, ab.Get(k-j+ka1-1, j-1)-bb.Get(k-i+kb1-1, i-1)*ab.Get(i-j+ka1-1, j-1))
				}
			}

			if wantx {
				//              post-multiply X by inv(S(i))
				goblas.Zdscal(toPtr((*n)-m), toPtrf64(one/bii), x.CVector(m+1-1, i-1), func() *int { y := 1; return &y }())
				if kbt > 0 {
					goblas.Zgerc(toPtr((*n)-m), &kbt, toPtrc128(-cone), x.CVector(m+1-1, i-1), func() *int { y := 1; return &y }(), bb.CVector(kb1-kbt-1, i-1), func() *int { y := 1; return &y }(), x.Off(m+1-1, i-kbt-1), ldx)
				}
			}

			//           store a(i,i1) in RA1 for use in next loop over K
			ra1 = ab.Get(i-i1+ka1-1, i1-1)
		}

		//        Generate and apply vectors of rotations to chase all the
		//        existing bulges KA positions down toward the bottom of the
		//        band
		for k = 1; k <= (*kb)-1; k++ {
			if update {
				//              Determine the rotations which would annihilate the bulge
				//              which has in theory just been created
				if i-k+(*ka) < (*n) && i-k > 1 {
					//                 generate rotation to annihilate a(i,i-k+ka+1)
					Zlartg(ab.GetPtr(k+1-1, i-k+(*ka)-1), &ra1, rwork.GetPtr(i-k+(*ka)-m-1), work.GetPtr(i-k+(*ka)-m-1), &ra)

					//                 create nonzero element a(i-k,i-k+ka+1) outside the
					//                 band and store it in WORK(i-k)
					t = -bb.Get(kb1-k-1, i-1) * ra1
					work.Set(i-k-1, rwork.GetCmplx(i-k+(*ka)-m-1)*t-work.GetConj(i-k+(*ka)-m-1)*ab.Get(0, i-k+(*ka)-1))
					ab.Set(0, i-k+(*ka)-1, work.Get(i-k+(*ka)-m-1)*t+rwork.GetCmplx(i-k+(*ka)-m-1)*ab.Get(0, i-k+(*ka)-1))
					ra1 = ra
				}
			}
			j2 = i - k - 1 + maxint(1, k-i0+2)*ka1
			nr = ((*n) - j2 + (*ka)) / ka1
			j1 = j2 + (nr-1)*ka1
			if update {
				j2t = maxint(j2, i+2*(*ka)-k+1)
			} else {
				j2t = j2
			}
			nrt = ((*n) - j2t + (*ka)) / ka1
			for j = j2t; j <= j1; j += ka1 {
				//              create nonzero element a(j-ka,j+1) outside the band
				//              and store it in WORK(j-m)
				work.Set(j-m-1, work.Get(j-m-1)*ab.Get(0, j+1-1))
				ab.Set(0, j+1-1, rwork.GetCmplx(j-m-1)*ab.Get(0, j+1-1))
			}

			//           generate rotations in 1st set to annihilate elements which
			//           have been created outside the band
			if nrt > 0 {
				Zlargv(&nrt, ab.CVector(0, j2t-1), &inca, work.Off(j2t-m-1), &ka1, rwork.Off(j2t-m-1), &ka1)
			}
			if nr > 0 {
				//              apply rotations in 1st set from the right
				for l = 1; l <= (*ka)-1; l++ {
					Zlartv(&nr, ab.CVector(ka1-l-1, j2-1), &inca, ab.CVector((*ka)-l-1, j2+1-1), &inca, rwork.Off(j2-m-1), work.Off(j2-m-1), &ka1)
				}

				//              apply rotations in 1st set from both sides to diagonal
				//              blocks
				Zlar2v(&nr, ab.CVector(ka1-1, j2-1), ab.CVector(ka1-1, j2+1-1), ab.CVector((*ka)-1, j2+1-1), &inca, rwork.Off(j2-m-1), work.Off(j2-m-1), &ka1)

				Zlacgv(&nr, work.Off(j2-m-1), &ka1)
			}

			//           start applying rotations in 1st set from the left
			for l = (*ka) - 1; l >= (*kb)-k+1; l-- {
				nrt = ((*n) - j2 + l) / ka1
				if nrt > 0 {
					Zlartv(&nrt, ab.CVector(l-1, j2+ka1-l-1), &inca, ab.CVector(l+1-1, j2+ka1-l-1), &inca, rwork.Off(j2-m-1), work.Off(j2-m-1), &ka1)
				}
			}

			if wantx {
				//              post-multiply X by product of rotations in 1st set
				for j = j2; j <= j1; j += ka1 {
					Zrot(toPtr((*n)-m), x.CVector(m+1-1, j-1), func() *int { y := 1; return &y }(), x.CVector(m+1-1, j+1-1), func() *int { y := 1; return &y }(), rwork.GetPtr(j-m-1), toPtrc128(work.GetConj(j-m-1)))
				}
			}
		}

		if update {
			if i2 <= (*n) && kbt > 0 {
				//              create nonzero element a(i-kbt,i-kbt+ka+1) outside the
				//              band and store it in WORK(i-kbt)
				work.Set(i-kbt-1, -bb.Get(kb1-kbt-1, i-1)*ra1)
			}
		}

		for k = (*kb); k >= 1; k-- {
			if update {
				j2 = i - k - 1 + maxint(2, k-i0+1)*ka1
			} else {
				j2 = i - k - 1 + maxint(1, k-i0+1)*ka1
			}

			//           finish applying rotations in 2nd set from the left
			for l = (*kb) - k; l >= 1; l-- {
				nrt = ((*n) - j2 + (*ka) + l) / ka1
				if nrt > 0 {
					Zlartv(&nrt, ab.CVector(l-1, j2-l+1-1), &inca, ab.CVector(l+1-1, j2-l+1-1), &inca, rwork.Off(j2-(*ka)-1), work.Off(j2-(*ka)-1), &ka1)
				}
			}
			nr = ((*n) - j2 + (*ka)) / ka1
			j1 = j2 + (nr-1)*ka1
			for j = j1; j >= j2; j -= ka1 {
				work.Set(j-1, work.Get(j-(*ka)-1))
				rwork.Set(j-1, rwork.Get(j-(*ka)-1))
			}
			for j = j2; j <= j1; j += ka1 {
				//              create nonzero element a(j-ka,j+1) outside the band
				//              and store it in WORK(j)
				work.Set(j-1, work.Get(j-1)*ab.Get(0, j+1-1))
				ab.Set(0, j+1-1, rwork.GetCmplx(j-1)*ab.Get(0, j+1-1))
			}
			if update {
				if i-k < (*n)-(*ka) && k <= kbt {
					work.Set(i-k+(*ka)-1, work.Get(i-k-1))
				}
			}
		}

		for k = (*kb); k >= 1; k-- {
			j2 = i - k - 1 + maxint(1, k-i0+1)*ka1
			nr = ((*n) - j2 + (*ka)) / ka1
			j1 = j2 + (nr-1)*ka1
			if nr > 0 {
				//              generate rotations in 2nd set to annihilate elements
				//              which have been created outside the band
				Zlargv(&nr, ab.CVector(0, j2-1), &inca, work.Off(j2-1), &ka1, rwork.Off(j2-1), &ka1)

				//              apply rotations in 2nd set from the right
				for l = 1; l <= (*ka)-1; l++ {
					Zlartv(&nr, ab.CVector(ka1-l-1, j2-1), &inca, ab.CVector((*ka)-l-1, j2+1-1), &inca, rwork.Off(j2-1), work.Off(j2-1), &ka1)
				}

				//              apply rotations in 2nd set from both sides to diagonal
				//              blocks
				Zlar2v(&nr, ab.CVector(ka1-1, j2-1), ab.CVector(ka1-1, j2+1-1), ab.CVector((*ka)-1, j2+1-1), &inca, rwork.Off(j2-1), work.Off(j2-1), &ka1)

				Zlacgv(&nr, work.Off(j2-1), &ka1)
			}

			//           start applying rotations in 2nd set from the left
			for l = (*ka) - 1; l >= (*kb)-k+1; l-- {
				nrt = ((*n) - j2 + l) / ka1
				if nrt > 0 {
					Zlartv(&nrt, ab.CVector(l-1, j2+ka1-l-1), &inca, ab.CVector(l+1-1, j2+ka1-l-1), &inca, rwork.Off(j2-1), work.Off(j2-1), &ka1)
				}
			}

			if wantx {
				//              post-multiply X by product of rotations in 2nd set
				for j = j2; j <= j1; j += ka1 {
					Zrot(toPtr((*n)-m), x.CVector(m+1-1, j-1), func() *int { y := 1; return &y }(), x.CVector(m+1-1, j+1-1), func() *int { y := 1; return &y }(), rwork.GetPtr(j-1), toPtrc128(work.GetConj(j-1)))
				}
			}
		}

		for k = 1; k <= (*kb)-1; k++ {
			j2 = i - k - 1 + maxint(1, k-i0+2)*ka1

			//           finish applying rotations in 1st set from the left
			for l = (*kb) - k; l >= 1; l-- {
				nrt = ((*n) - j2 + l) / ka1
				if nrt > 0 {
					Zlartv(&nrt, ab.CVector(l-1, j2+ka1-l-1), &inca, ab.CVector(l+1-1, j2+ka1-l-1), &inca, rwork.Off(j2-m-1), work.Off(j2-m-1), &ka1)
				}
			}
		}

		if (*kb) > 1 {
			for j = (*n) - 1; j >= j2+(*ka); j-- {
				rwork.Set(j-m-1, rwork.Get(j-(*ka)-m-1))
				work.Set(j-m-1, work.Get(j-(*ka)-m-1))
			}
		}

	} else {
		//        Transform A, working with the lower triangle
		if update {
			//           Form  inv(S(i))**H * A * inv(S(i))
			bii = bb.GetRe(0, i-1)
			ab.SetRe(0, i-1, (ab.GetRe(0, i-1)/bii)/bii)
			for j = i + 1; j <= i1; j++ {
				ab.Set(j-i+1-1, i-1, ab.Get(j-i+1-1, i-1)/complex(bii, 0))
			}
			for j = maxint(1, i-(*ka)); j <= i-1; j++ {
				ab.Set(i-j+1-1, j-1, ab.Get(i-j+1-1, j-1)/complex(bii, 0))
			}
			for k = i - kbt; k <= i-1; k++ {
				for j = i - kbt; j <= k; j++ {
					ab.Set(k-j+1-1, j-1, ab.Get(k-j+1-1, j-1)-bb.Get(i-j+1-1, j-1)*ab.GetConj(i-k+1-1, k-1)-bb.GetConj(i-k+1-1, k-1)*ab.Get(i-j+1-1, j-1)+ab.GetConj(0, i-1)*bb.Get(i-j+1-1, j-1)*bb.GetConj(i-k+1-1, k-1))
				}
				for j = maxint(1, i-(*ka)); j <= i-kbt-1; j++ {
					ab.Set(k-j+1-1, j-1, ab.Get(k-j+1-1, j-1)-bb.GetConj(i-k+1-1, k-1)*ab.Get(i-j+1-1, j-1))
				}
			}
			for j = i; j <= i1; j++ {
				for k = maxint(j-(*ka), i-kbt); k <= i-1; k++ {
					ab.Set(j-k+1-1, k-1, ab.Get(j-k+1-1, k-1)-bb.Get(i-k+1-1, k-1)*ab.Get(j-i+1-1, i-1))
				}
			}

			if wantx {
				//              post-multiply X by inv(S(i))
				goblas.Zdscal(toPtr((*n)-m), toPtrf64(one/bii), x.CVector(m+1-1, i-1), func() *int { y := 1; return &y }())
				if kbt > 0 {
					goblas.Zgeru(toPtr((*n)-m), &kbt, toPtrc128(-cone), x.CVector(m+1-1, i-1), func() *int { y := 1; return &y }(), bb.CVector(kbt+1-1, i-kbt-1), toPtr((*ldbb)-1), x.Off(m+1-1, i-kbt-1), ldx)
				}
			}

			//           store a(i1,i) in RA1 for use in next loop over K
			ra1 = ab.Get(i1-i+1-1, i-1)
		}

		//        Generate and apply vectors of rotations to chase all the
		//        existing bulges KA positions down toward the bottom of the
		//        band
		for k = 1; k <= (*kb)-1; k++ {
			if update {
				//              Determine the rotations which would annihilate the bulge
				//              which has in theory just been created
				if i-k+(*ka) < (*n) && i-k > 1 {
					//                 generate rotation to annihilate a(i-k+ka+1,i)
					Zlartg(ab.GetPtr(ka1-k-1, i-1), &ra1, rwork.GetPtr(i-k+(*ka)-m-1), work.GetPtr(i-k+(*ka)-m-1), &ra)

					//                 create nonzero element a(i-k+ka+1,i-k) outside the
					//                 band and store it in WORK(i-k)
					t = -bb.Get(k+1-1, i-k-1) * ra1
					work.Set(i-k-1, rwork.GetCmplx(i-k+(*ka)-m-1)*t-work.GetConj(i-k+(*ka)-m-1)*ab.Get(ka1-1, i-k-1))
					ab.Set(ka1-1, i-k-1, work.Get(i-k+(*ka)-m-1)*t+rwork.GetCmplx(i-k+(*ka)-m-1)*ab.Get(ka1-1, i-k-1))
					ra1 = ra
				}
			}
			j2 = i - k - 1 + maxint(1, k-i0+2)*ka1
			nr = ((*n) - j2 + (*ka)) / ka1
			j1 = j2 + (nr-1)*ka1
			if update {
				j2t = maxint(j2, i+2*(*ka)-k+1)
			} else {
				j2t = j2
			}
			nrt = ((*n) - j2t + (*ka)) / ka1
			for j = j2t; j <= j1; j += ka1 {
				//              create nonzero element a(j+1,j-ka) outside the band
				//              and store it in WORK(j-m)
				work.Set(j-m-1, work.Get(j-m-1)*ab.Get(ka1-1, j-(*ka)+1-1))
				ab.Set(ka1-1, j-(*ka)+1-1, rwork.GetCmplx(j-m-1)*ab.Get(ka1-1, j-(*ka)+1-1))
			}

			//           generate rotations in 1st set to annihilate elements which
			//           have been created outside the band
			if nrt > 0 {
				Zlargv(&nrt, ab.CVector(ka1-1, j2t-(*ka)-1), &inca, work.Off(j2t-m-1), &ka1, rwork.Off(j2t-m-1), &ka1)
			}
			if nr > 0 {
				//              apply rotations in 1st set from the left
				for l = 1; l <= (*ka)-1; l++ {
					Zlartv(&nr, ab.CVector(l+1-1, j2-l-1), &inca, ab.CVector(l+2-1, j2-l-1), &inca, rwork.Off(j2-m-1), work.Off(j2-m-1), &ka1)
				}

				//              apply rotations in 1st set from both sides to diagonal
				//              blocks
				Zlar2v(&nr, ab.CVector(0, j2-1), ab.CVector(0, j2+1-1), ab.CVector(1, j2-1), &inca, rwork.Off(j2-m-1), work.Off(j2-m-1), &ka1)

				Zlacgv(&nr, work.Off(j2-m-1), &ka1)
			}

			//           start applying rotations in 1st set from the right
			for l = (*ka) - 1; l >= (*kb)-k+1; l-- {
				nrt = ((*n) - j2 + l) / ka1
				if nrt > 0 {
					Zlartv(&nrt, ab.CVector(ka1-l+1-1, j2-1), &inca, ab.CVector(ka1-l-1, j2+1-1), &inca, rwork.Off(j2-m-1), work.Off(j2-m-1), &ka1)
				}
			}

			if wantx {
				//              post-multiply X by product of rotations in 1st set
				for j = j2; j <= j1; j += ka1 {
					Zrot(toPtr((*n)-m), x.CVector(m+1-1, j-1), func() *int { y := 1; return &y }(), x.CVector(m+1-1, j+1-1), func() *int { y := 1; return &y }(), rwork.GetPtr(j-m-1), work.GetPtr(j-m-1))
				}
			}
		}

		if update {
			if i2 <= (*n) && kbt > 0 {
				//              create nonzero element a(i-kbt+ka+1,i-kbt) outside the
				//              band and store it in WORK(i-kbt)
				work.Set(i-kbt-1, -bb.Get(kbt+1-1, i-kbt-1)*ra1)
			}
		}

		for k = (*kb); k >= 1; k-- {
			if update {
				j2 = i - k - 1 + maxint(2, k-i0+1)*ka1
			} else {
				j2 = i - k - 1 + maxint(1, k-i0+1)*ka1
			}

			//           finish applying rotations in 2nd set from the right
			for l = (*kb) - k; l >= 1; l-- {
				nrt = ((*n) - j2 + (*ka) + l) / ka1
				if nrt > 0 {
					Zlartv(&nrt, ab.CVector(ka1-l+1-1, j2-(*ka)-1), &inca, ab.CVector(ka1-l-1, j2-(*ka)+1-1), &inca, rwork.Off(j2-(*ka)-1), work.Off(j2-(*ka)-1), &ka1)
				}
			}
			nr = ((*n) - j2 + (*ka)) / ka1
			j1 = j2 + (nr-1)*ka1
			for j = j1; j >= j2; j -= ka1 {
				work.Set(j-1, work.Get(j-(*ka)-1))
				rwork.Set(j-1, rwork.Get(j-(*ka)-1))
			}
			for j = j2; j <= j1; j += ka1 {
				//              create nonzero element a(j+1,j-ka) outside the band
				//              and store it in WORK(j)
				work.Set(j-1, work.Get(j-1)*ab.Get(ka1-1, j-(*ka)+1-1))
				ab.Set(ka1-1, j-(*ka)+1-1, rwork.GetCmplx(j-1)*ab.Get(ka1-1, j-(*ka)+1-1))
			}
			if update {
				if i-k < (*n)-(*ka) && k <= kbt {
					work.Set(i-k+(*ka)-1, work.Get(i-k-1))
				}
			}
		}

		for k = (*kb); k >= 1; k-- {
			j2 = i - k - 1 + maxint(1, k-i0+1)*ka1
			nr = ((*n) - j2 + (*ka)) / ka1
			j1 = j2 + (nr-1)*ka1
			if nr > 0 {
				//              generate rotations in 2nd set to annihilate elements
				//              which have been created outside the band
				Zlargv(&nr, ab.CVector(ka1-1, j2-(*ka)-1), &inca, work.Off(j2-1), &ka1, rwork.Off(j2-1), &ka1)

				//              apply rotations in 2nd set from the left
				for l = 1; l <= (*ka)-1; l++ {
					Zlartv(&nr, ab.CVector(l+1-1, j2-l-1), &inca, ab.CVector(l+2-1, j2-l-1), &inca, rwork.Off(j2-1), work.Off(j2-1), &ka1)
				}

				//              apply rotations in 2nd set from both sides to diagonal
				//              blocks
				Zlar2v(&nr, ab.CVector(0, j2-1), ab.CVector(0, j2+1-1), ab.CVector(1, j2-1), &inca, rwork.Off(j2-1), work.Off(j2-1), &ka1)

				Zlacgv(&nr, work.Off(j2-1), &ka1)
			}

			//           start applying rotations in 2nd set from the right
			for l = (*ka) - 1; l >= (*kb)-k+1; l-- {
				nrt = ((*n) - j2 + l) / ka1
				if nrt > 0 {
					Zlartv(&nrt, ab.CVector(ka1-l+1-1, j2-1), &inca, ab.CVector(ka1-l-1, j2+1-1), &inca, rwork.Off(j2-1), work.Off(j2-1), &ka1)
				}
			}

			if wantx {
				//              post-multiply X by product of rotations in 2nd set
				for j = j2; j <= j1; j += ka1 {
					Zrot(toPtr((*n)-m), x.CVector(m+1-1, j-1), func() *int { y := 1; return &y }(), x.CVector(m+1-1, j+1-1), func() *int { y := 1; return &y }(), rwork.GetPtr(j-1), work.GetPtr(j-1))
				}
			}
		}

		for k = 1; k <= (*kb)-1; k++ {
			j2 = i - k - 1 + maxint(1, k-i0+2)*ka1

			//           finish applying rotations in 1st set from the right
			for l = (*kb) - k; l >= 1; l-- {
				nrt = ((*n) - j2 + l) / ka1
				if nrt > 0 {
					Zlartv(&nrt, ab.CVector(ka1-l+1-1, j2-1), &inca, ab.CVector(ka1-l-1, j2+1-1), &inca, rwork.Off(j2-m-1), work.Off(j2-m-1), &ka1)
				}
			}
		}

		if (*kb) > 1 {
			for j = (*n) - 1; j >= j2+(*ka); j-- {
				rwork.Set(j-m-1, rwork.Get(j-(*ka)-m-1))
				work.Set(j-m-1, work.Get(j-(*ka)-m-1))
			}
		}

	}

	goto label10

label480:
	;

	//     **************************** Phase 2 *****************************
	//
	//     The logical structure of this phase is:
	//
	//     UPDATE = .TRUE.
	//     DO I = 1, M
	//        use S(i) to update A and create a new bulge
	//        apply rotations to push all bulges KA positions upward
	//     END DO
	//     UPDATE = .FALSE.
	//     DO I = M - KA - 1, 2, -1
	//        apply rotations to push all bulges KA positions upward
	//     END DO
	//
	//     To avoid duplicating code, the two loops are merged.
	update = true
	i = 0
label490:
	;
	if update {
		i = i + 1
		kbt = minint(*kb, m-i)
		i0 = i + 1
		i1 = maxint(1, i-(*ka))
		i2 = i + kbt - ka1
		if i > m {
			update = false
			i = i - 1
			i0 = m + 1
			if (*ka) == 0 {
				return
			}
			goto label490
		}
	} else {
		i = i - (*ka)
		if i < 2 {
			return
		}
	}

	if i < m-kbt {
		nx = m
	} else {
		nx = (*n)
	}

	if upper {
		//        Transform A, working with the upper triangle
		if update {
			//           Form  inv(S(i))**H * A * inv(S(i))
			bii = bb.GetRe(kb1-1, i-1)
			ab.SetRe(ka1-1, i-1, (ab.GetRe(ka1-1, i-1)/bii)/bii)
			for j = i1; j <= i-1; j++ {
				ab.Set(j-i+ka1-1, i-1, ab.Get(j-i+ka1-1, i-1)/complex(bii, 0))
			}
			for j = i + 1; j <= minint(*n, i+(*ka)); j++ {
				ab.Set(i-j+ka1-1, j-1, ab.Get(i-j+ka1-1, j-1)/complex(bii, 0))
			}
			for k = i + 1; k <= i+kbt; k++ {
				for j = k; j <= i+kbt; j++ {
					ab.Set(k-j+ka1-1, j-1, ab.Get(k-j+ka1-1, j-1)-bb.Get(i-j+kb1-1, j-1)*ab.GetConj(i-k+ka1-1, k-1)-bb.GetConj(i-k+kb1-1, k-1)*ab.Get(i-j+ka1-1, j-1)+ab.GetReCmplx(ka1-1, i-1)*bb.Get(i-j+kb1-1, j-1)*bb.GetConj(i-k+kb1-1, k-1))
				}
				for j = i + kbt + 1; j <= minint(*n, i+(*ka)); j++ {
					ab.Set(k-j+ka1-1, j-1, ab.Get(k-j+ka1-1, j-1)-bb.GetConj(i-k+kb1-1, k-1)*ab.Get(i-j+ka1-1, j-1))
				}
			}
			for j = i1; j <= i; j++ {
				for k = i + 1; k <= minint(j+(*ka), i+kbt); k++ {
					ab.Set(j-k+ka1-1, k-1, ab.Get(j-k+ka1-1, k-1)-bb.Get(i-k+kb1-1, k-1)*ab.Get(j-i+ka1-1, i-1))
				}
			}

			if wantx {
				//              post-multiply X by inv(S(i))
				goblas.Zdscal(&nx, toPtrf64(one/bii), x.CVector(0, i-1), func() *int { y := 1; return &y }())
				if kbt > 0 {
					goblas.Zgeru(&nx, &kbt, toPtrc128(-cone), x.CVector(0, i-1), func() *int { y := 1; return &y }(), bb.CVector((*kb)-1, i+1-1), toPtr((*ldbb)-1), x.Off(0, i+1-1), ldx)
				}
			}

			//           store a(i1,i) in RA1 for use in next loop over K
			ra1 = ab.Get(i1-i+ka1-1, i-1)
		}

		//        Generate and apply vectors of rotations to chase all the
		//        existing bulges KA positions up toward the top of the band
		for k = 1; k <= (*kb)-1; k++ {
			if update {
				//              Determine the rotations which would annihilate the bulge
				//              which has in theory just been created
				if i+k-ka1 > 0 && i+k < m {
					//                 generate rotation to annihilate a(i+k-ka-1,i)
					Zlartg(ab.GetPtr(k+1-1, i-1), &ra1, rwork.GetPtr(i+k-(*ka)-1), work.GetPtr(i+k-(*ka)-1), &ra)

					//                 create nonzero element a(i+k-ka-1,i+k) outside the
					//                 band and store it in WORK(m-kb+i+k)
					t = -bb.Get(kb1-k-1, i+k-1) * ra1
					work.Set(m-(*kb)+i+k-1, rwork.GetCmplx(i+k-(*ka)-1)*t-work.GetConj(i+k-(*ka)-1)*ab.Get(0, i+k-1))
					ab.Set(0, i+k-1, work.Get(i+k-(*ka)-1)*t+rwork.GetCmplx(i+k-(*ka)-1)*ab.Get(0, i+k-1))
					ra1 = ra
				}
			}
			j2 = i + k + 1 - maxint(1, k+i0-m+1)*ka1
			nr = (j2 + (*ka) - 1) / ka1
			j1 = j2 - (nr-1)*ka1
			if update {
				j2t = minint(j2, i-2*(*ka)+k-1)
			} else {
				j2t = j2
			}
			nrt = (j2t + (*ka) - 1) / ka1
			for j = j1; j <= j2t; j += ka1 {
				//              create nonzero element a(j-1,j+ka) outside the band
				//              and store it in WORK(j)
				work.Set(j-1, work.Get(j-1)*ab.Get(0, j+(*ka)-1-1))
				ab.Set(0, j+(*ka)-1-1, rwork.GetCmplx(j-1)*ab.Get(0, j+(*ka)-1-1))
			}

			//           generate rotations in 1st set to annihilate elements which
			//           have been created outside the band
			if nrt > 0 {
				Zlargv(&nrt, ab.CVector(0, j1+(*ka)-1), &inca, work.Off(j1-1), &ka1, rwork.Off(j1-1), &ka1)
			}
			if nr > 0 {
				//              apply rotations in 1st set from the left
				for l = 1; l <= (*ka)-1; l++ {
					Zlartv(&nr, ab.CVector(ka1-l-1, j1+l-1), &inca, ab.CVector((*ka)-l-1, j1+l-1), &inca, rwork.Off(j1-1), work.Off(j1-1), &ka1)
				}

				//              apply rotations in 1st set from both sides to diagonal
				//              blocks
				Zlar2v(&nr, ab.CVector(ka1-1, j1-1), ab.CVector(ka1-1, j1-1-1), ab.CVector((*ka)-1, j1-1), &inca, rwork.Off(j1-1), work.Off(j1-1), &ka1)

				Zlacgv(&nr, work.Off(j1-1), &ka1)
			}

			//           start applying rotations in 1st set from the right
			for l = (*ka) - 1; l >= (*kb)-k+1; l-- {
				nrt = (j2 + l - 1) / ka1
				j1t = j2 - (nrt-1)*ka1
				if nrt > 0 {
					Zlartv(&nrt, ab.CVector(l-1, j1t-1), &inca, ab.CVector(l+1-1, j1t-1-1), &inca, rwork.Off(j1t-1), work.Off(j1t-1), &ka1)
				}
			}

			if wantx {
				//              post-multiply X by product of rotations in 1st set
				for j = j1; j <= j2; j += ka1 {
					Zrot(&nx, x.CVector(0, j-1), func() *int { y := 1; return &y }(), x.CVector(0, j-1-1), func() *int { y := 1; return &y }(), rwork.GetPtr(j-1), work.GetPtr(j-1))
				}
			}
		}

		if update {
			if i2 > 0 && kbt > 0 {
				//              create nonzero element a(i+kbt-ka-1,i+kbt) outside the
				//              band and store it in WORK(m-kb+i+kbt)
				work.Set(m-(*kb)+i+kbt-1, -bb.Get(kb1-kbt-1, i+kbt-1)*ra1)
			}
		}

		for k = (*kb); k >= 1; k-- {
			if update {
				j2 = i + k + 1 - maxint(2, k+i0-m)*ka1
			} else {
				j2 = i + k + 1 - maxint(1, k+i0-m)*ka1
			}

			//           finish applying rotations in 2nd set from the right
			for l = (*kb) - k; l >= 1; l-- {
				nrt = (j2 + (*ka) + l - 1) / ka1
				j1t = j2 - (nrt-1)*ka1
				if nrt > 0 {
					Zlartv(&nrt, ab.CVector(l-1, j1t+(*ka)-1), &inca, ab.CVector(l+1-1, j1t+(*ka)-1-1), &inca, rwork.Off(m-(*kb)+j1t+(*ka)-1), work.Off(m-(*kb)+j1t+(*ka)-1), &ka1)
				}
			}
			nr = (j2 + (*ka) - 1) / ka1
			j1 = j2 - (nr-1)*ka1
			for j = j1; j <= j2; j += ka1 {
				work.Set(m-(*kb)+j-1, work.Get(m-(*kb)+j+(*ka)-1))
				rwork.Set(m-(*kb)+j-1, rwork.Get(m-(*kb)+j+(*ka)-1))
			}
			for j = j1; j <= j2; j += ka1 {
				//              create nonzero element a(j-1,j+ka) outside the band
				//              and store it in WORK(m-kb+j)
				work.Set(m-(*kb)+j-1, work.Get(m-(*kb)+j-1)*ab.Get(0, j+(*ka)-1-1))
				ab.Set(0, j+(*ka)-1-1, rwork.GetCmplx(m-(*kb)+j-1)*ab.Get(0, j+(*ka)-1-1))
			}
			if update {
				if i+k > ka1 && k <= kbt {
					work.Set(m-(*kb)+i+k-(*ka)-1, work.Get(m-(*kb)+i+k-1))
				}
			}
		}

		for k = (*kb); k >= 1; k-- {
			j2 = i + k + 1 - maxint(1, k+i0-m)*ka1
			nr = (j2 + (*ka) - 1) / ka1
			j1 = j2 - (nr-1)*ka1
			if nr > 0 {
				//              generate rotations in 2nd set to annihilate elements
				//              which have been created outside the band
				Zlargv(&nr, ab.CVector(0, j1+(*ka)-1), &inca, work.Off(m-(*kb)+j1-1), &ka1, rwork.Off(m-(*kb)+j1-1), &ka1)

				//              apply rotations in 2nd set from the left
				for l = 1; l <= (*ka)-1; l++ {
					Zlartv(&nr, ab.CVector(ka1-l-1, j1+l-1), &inca, ab.CVector((*ka)-l-1, j1+l-1), &inca, rwork.Off(m-(*kb)+j1-1), work.Off(m-(*kb)+j1-1), &ka1)
				}

				//              apply rotations in 2nd set from both sides to diagonal
				//              blocks
				Zlar2v(&nr, ab.CVector(ka1-1, j1-1), ab.CVector(ka1-1, j1-1-1), ab.CVector((*ka)-1, j1-1), &inca, rwork.Off(m-(*kb)+j1-1), work.Off(m-(*kb)+j1-1), &ka1)

				Zlacgv(&nr, work.Off(m-(*kb)+j1-1), &ka1)
			}

			//           start applying rotations in 2nd set from the right
			for l = (*ka) - 1; l >= (*kb)-k+1; l-- {
				nrt = (j2 + l - 1) / ka1
				j1t = j2 - (nrt-1)*ka1
				if nrt > 0 {
					Zlartv(&nrt, ab.CVector(l-1, j1t-1), &inca, ab.CVector(l+1-1, j1t-1-1), &inca, rwork.Off(m-(*kb)+j1t-1), work.Off(m-(*kb)+j1t-1), &ka1)
				}
			}

			if wantx {
				//              post-multiply X by product of rotations in 2nd set
				for j = j1; j <= j2; j += ka1 {
					Zrot(&nx, x.CVector(0, j-1), func() *int { y := 1; return &y }(), x.CVector(0, j-1-1), func() *int { y := 1; return &y }(), rwork.GetPtr(m-(*kb)+j-1), work.GetPtr(m-(*kb)+j-1))
				}
			}
		}

		for k = 1; k <= (*kb)-1; k++ {
			j2 = i + k + 1 - maxint(1, k+i0-m+1)*ka1

			//           finish applying rotations in 1st set from the right
			for l = (*kb) - k; l >= 1; l-- {
				nrt = (j2 + l - 1) / ka1
				j1t = j2 - (nrt-1)*ka1
				if nrt > 0 {
					Zlartv(&nrt, ab.CVector(l-1, j1t-1), &inca, ab.CVector(l+1-1, j1t-1-1), &inca, rwork.Off(j1t-1), work.Off(j1t-1), &ka1)
				}
			}
		}

		if (*kb) > 1 {
			for j = 2; j <= i2-(*ka); j++ {
				rwork.Set(j-1, rwork.Get(j+(*ka)-1))
				work.Set(j-1, work.Get(j+(*ka)-1))
			}
		}

	} else {
		//        Transform A, working with the lower triangle
		if update {
			//           Form  inv(S(i))**H * A * inv(S(i))
			bii = bb.GetRe(0, i-1)
			ab.SetRe(0, i-1, (ab.GetRe(0, i-1)/bii)/bii)
			for j = i1; j <= i-1; j++ {
				ab.Set(i-j+1-1, j-1, ab.Get(i-j+1-1, j-1)/complex(bii, 0))
			}
			for j = i + 1; j <= minint(*n, i+(*ka)); j++ {
				ab.Set(j-i+1-1, i-1, ab.Get(j-i+1-1, i-1)/complex(bii, 0))
			}
			for k = i + 1; k <= i+kbt; k++ {
				for j = k; j <= i+kbt; j++ {
					ab.Set(j-k+1-1, k-1, ab.Get(j-k+1-1, k-1)-bb.Get(j-i+1-1, i-1)*ab.GetConj(k-i+1-1, i-1)-bb.GetConj(k-i+1-1, i-1)*ab.Get(j-i+1-1, i-1)+ab.GetReCmplx(0, i-1)*bb.Get(j-i+1-1, i-1)*bb.GetConj(k-i+1-1, i-1))
				}
				for j = i + kbt + 1; j <= minint(*n, i+(*ka)); j++ {
					ab.Set(j-k+1-1, k-1, ab.Get(j-k+1-1, k-1)-bb.GetConj(k-i+1-1, i-1)*ab.Get(j-i+1-1, i-1))
				}
			}
			for j = i1; j <= i; j++ {
				for k = i + 1; k <= minint(j+(*ka), i+kbt); k++ {
					ab.Set(k-j+1-1, j-1, ab.Get(k-j+1-1, j-1)-bb.Get(k-i+1-1, i-1)*ab.Get(i-j+1-1, j-1))
				}
			}

			if wantx {
				//              post-multiply X by inv(S(i))
				goblas.Zdscal(&nx, toPtrf64(one/bii), x.CVector(0, i-1), func() *int { y := 1; return &y }())
				if kbt > 0 {
					goblas.Zgerc(&nx, &kbt, toPtrc128(-cone), x.CVector(0, i-1), func() *int { y := 1; return &y }(), bb.CVector(1, i-1), func() *int { y := 1; return &y }(), x.Off(0, i+1-1), ldx)
				}
			}

			//           store a(i,i1) in RA1 for use in next loop over K
			ra1 = ab.Get(i-i1+1-1, i1-1)
		}

		//        Generate and apply vectors of rotations to chase all the
		//        existing bulges KA positions up toward the top of the band
		for k = 1; k <= (*kb)-1; k++ {
			if update {
				//              Determine the rotations which would annihilate the bulge
				//              which has in theory just been created
				if i+k-ka1 > 0 && i+k < m {
					//                 generate rotation to annihilate a(i,i+k-ka-1)
					Zlartg(ab.GetPtr(ka1-k-1, i+k-(*ka)-1), &ra1, rwork.GetPtr(i+k-(*ka)-1), work.GetPtr(i+k-(*ka)-1), &ra)

					//                 create nonzero element a(i+k,i+k-ka-1) outside the
					//                 band and store it in WORK(m-kb+i+k)
					t = -bb.Get(k+1-1, i-1) * ra1
					work.Set(m-(*kb)+i+k-1, rwork.GetCmplx(i+k-(*ka)-1)*t-work.GetConj(i+k-(*ka)-1)*ab.Get(ka1-1, i+k-(*ka)-1))
					ab.Set(ka1-1, i+k-(*ka)-1, work.Get(i+k-(*ka)-1)*t+rwork.GetCmplx(i+k-(*ka)-1)*ab.Get(ka1-1, i+k-(*ka)-1))
					ra1 = ra
				}
			}
			j2 = i + k + 1 - maxint(1, k+i0-m+1)*ka1
			nr = (j2 + (*ka) - 1) / ka1
			j1 = j2 - (nr-1)*ka1
			if update {
				j2t = minint(j2, i-2*(*ka)+k-1)
			} else {
				j2t = j2
			}
			nrt = (j2t + (*ka) - 1) / ka1
			for j = j1; j <= j2t; j += ka1 {
				//              create nonzero element a(j+ka,j-1) outside the band
				//              and store it in WORK(j)
				work.Set(j-1, work.Get(j-1)*ab.Get(ka1-1, j-1-1))
				ab.Set(ka1-1, j-1-1, rwork.GetCmplx(j-1)*ab.Get(ka1-1, j-1-1))
			}

			//           generate rotations in 1st set to annihilate elements which
			//           have been created outside the band
			if nrt > 0 {
				Zlargv(&nrt, ab.CVector(ka1-1, j1-1), &inca, work.Off(j1-1), &ka1, rwork.Off(j1-1), &ka1)
			}
			if nr > 0 {
				//              apply rotations in 1st set from the right
				for l = 1; l <= (*ka)-1; l++ {
					Zlartv(&nr, ab.CVector(l+1-1, j1-1), &inca, ab.CVector(l+2-1, j1-1-1), &inca, rwork.Off(j1-1), work.Off(j1-1), &ka1)
				}

				//              apply rotations in 1st set from both sides to diagonal
				//              blocks
				Zlar2v(&nr, ab.CVector(0, j1-1), ab.CVector(0, j1-1-1), ab.CVector(1, j1-1-1), &inca, rwork.Off(j1-1), work.Off(j1-1), &ka1)

				Zlacgv(&nr, work.Off(j1-1), &ka1)
			}

			//           start applying rotations in 1st set from the left
			for l = (*ka) - 1; l >= (*kb)-k+1; l-- {
				nrt = (j2 + l - 1) / ka1
				j1t = j2 - (nrt-1)*ka1
				if nrt > 0 {
					Zlartv(&nrt, ab.CVector(ka1-l+1-1, j1t-ka1+l-1), &inca, ab.CVector(ka1-l-1, j1t-ka1+l-1), &inca, rwork.Off(j1t-1), work.Off(j1t-1), &ka1)
				}
			}

			if wantx {
				//              post-multiply X by product of rotations in 1st set
				for j = j1; j <= j2; j += ka1 {
					Zrot(&nx, x.CVector(0, j-1), func() *int { y := 1; return &y }(), x.CVector(0, j-1-1), func() *int { y := 1; return &y }(), rwork.GetPtr(j-1), toPtrc128(work.GetConj(j-1)))
				}
			}
		}

		if update {
			if i2 > 0 && kbt > 0 {
				//              create nonzero element a(i+kbt,i+kbt-ka-1) outside the
				//              band and store it in WORK(m-kb+i+kbt)
				work.Set(m-(*kb)+i+kbt-1, -bb.Get(kbt+1-1, i-1)*ra1)
			}
		}

		for k = (*kb); k >= 1; k-- {
			if update {
				j2 = i + k + 1 - maxint(2, k+i0-m)*ka1
			} else {
				j2 = i + k + 1 - maxint(1, k+i0-m)*ka1
			}

			//           finish applying rotations in 2nd set from the left
			for l = (*kb) - k; l >= 1; l-- {
				nrt = (j2 + (*ka) + l - 1) / ka1
				j1t = j2 - (nrt-1)*ka1
				if nrt > 0 {
					Zlartv(&nrt, ab.CVector(ka1-l+1-1, j1t+l-1-1), &inca, ab.CVector(ka1-l-1, j1t+l-1-1), &inca, rwork.Off(m-(*kb)+j1t+(*ka)-1), work.Off(m-(*kb)+j1t+(*ka)-1), &ka1)
				}
			}
			nr = (j2 + (*ka) - 1) / ka1
			j1 = j2 - (nr-1)*ka1
			for j = j1; j <= j2; j += ka1 {
				work.Set(m-(*kb)+j-1, work.Get(m-(*kb)+j+(*ka)-1))
				rwork.Set(m-(*kb)+j-1, rwork.Get(m-(*kb)+j+(*ka)-1))
			}
			for j = j1; j <= j2; j += ka1 {
				//              create nonzero element a(j+ka,j-1) outside the band
				//              and store it in WORK(m-kb+j)
				work.Set(m-(*kb)+j-1, work.Get(m-(*kb)+j-1)*ab.Get(ka1-1, j-1-1))
				ab.Set(ka1-1, j-1-1, rwork.GetCmplx(m-(*kb)+j-1)*ab.Get(ka1-1, j-1-1))
			}
			if update {
				if i+k > ka1 && k <= kbt {
					work.Set(m-(*kb)+i+k-(*ka)-1, work.Get(m-(*kb)+i+k-1))
				}
			}
		}

		for k = (*kb); k >= 1; k-- {
			j2 = i + k + 1 - maxint(1, k+i0-m)*ka1
			nr = (j2 + (*ka) - 1) / ka1
			j1 = j2 - (nr-1)*ka1
			if nr > 0 {
				//              generate rotations in 2nd set to annihilate elements
				//              which have been created outside the band
				Zlargv(&nr, ab.CVector(ka1-1, j1-1), &inca, work.Off(m-(*kb)+j1-1), &ka1, rwork.Off(m-(*kb)+j1-1), &ka1)

				//              apply rotations in 2nd set from the right
				for l = 1; l <= (*ka)-1; l++ {
					Zlartv(&nr, ab.CVector(l+1-1, j1-1), &inca, ab.CVector(l+2-1, j1-1-1), &inca, rwork.Off(m-(*kb)+j1-1), work.Off(m-(*kb)+j1-1), &ka1)
				}

				//              apply rotations in 2nd set from both sides to diagonal
				//              blocks
				Zlar2v(&nr, ab.CVector(0, j1-1), ab.CVector(0, j1-1-1), ab.CVector(1, j1-1-1), &inca, rwork.Off(m-(*kb)+j1-1), work.Off(m-(*kb)+j1-1), &ka1)

				Zlacgv(&nr, work.Off(m-(*kb)+j1-1), &ka1)
			}

			//           start applying rotations in 2nd set from the left
			for l = (*ka) - 1; l >= (*kb)-k+1; l-- {
				nrt = (j2 + l - 1) / ka1
				j1t = j2 - (nrt-1)*ka1
				if nrt > 0 {
					Zlartv(&nrt, ab.CVector(ka1-l+1-1, j1t-ka1+l-1), &inca, ab.CVector(ka1-l-1, j1t-ka1+l-1), &inca, rwork.Off(m-(*kb)+j1t-1), work.Off(m-(*kb)+j1t-1), &ka1)
				}
			}

			if wantx {
				//              post-multiply X by product of rotations in 2nd set
				for j = j1; j <= j2; j += ka1 {
					Zrot(&nx, x.CVector(0, j-1), func() *int { y := 1; return &y }(), x.CVector(0, j-1-1), func() *int { y := 1; return &y }(), rwork.GetPtr(m-(*kb)+j-1), toPtrc128(work.GetConj(m-(*kb)+j-1)))
				}
			}
		}

		for k = 1; k <= (*kb)-1; k++ {
			j2 = i + k + 1 - maxint(1, k+i0-m+1)*ka1

			//           finish applying rotations in 1st set from the left
			for l = (*kb) - k; l >= 1; l-- {
				nrt = (j2 + l - 1) / ka1
				j1t = j2 - (nrt-1)*ka1
				if nrt > 0 {
					Zlartv(&nrt, ab.CVector(ka1-l+1-1, j1t-ka1+l-1), &inca, ab.CVector(ka1-l-1, j1t-ka1+l-1), &inca, rwork.Off(j1t-1), work.Off(j1t-1), &ka1)
				}
			}
		}

		if (*kb) > 1 {
			for j = 2; j <= i2-(*ka); j++ {
				rwork.Set(j-1, rwork.Get(j+(*ka)-1))
				work.Set(j-1, work.Get(j+(*ka)-1))
			}
		}

	}

	goto label490
}
