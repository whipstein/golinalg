package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dsbgst reduces a real symmetric-definite banded generalized
// eigenproblem  A*x = lambda*B*x  to standard form  C*y = lambda*y,
// such that C has the same bandwidth as A.
//
// B must have been previously factorized as S**T*S by DPBSTF, using a
// split Cholesky factorization. A is overwritten by C = X**T*A*X, where
// X = S**(-1)*Q and Q is an orthogonal matrix chosen to preserve the
// bandwidth of A.
func Dsbgst(vect byte, uplo mat.MatUplo, n, ka, kb int, ab, bb, x *mat.Matrix, work *mat.Vector) (err error) {
	var update, upper, wantx bool
	var bii, one, ra, ra1, t, zero float64
	var i, i0, i1, i2, inca, j, j1, j1t, j2, j2t, k, ka1, kb1, kbt, l, m, nr, nrt, nx int

	zero = 0.0
	one = 1.0

	//     Test the input parameters
	wantx = vect == 'V'
	upper = uplo == Upper
	ka1 = ka + 1
	kb1 = kb + 1
	if !wantx && vect != 'N' {
		err = fmt.Errorf("!wantx && vect != 'N': vect='%c'", vect)
	} else if !upper && uplo != Lower {
		err = fmt.Errorf("!upper && uplo != Lower: uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if ka < 0 {
		err = fmt.Errorf("ka < 0: ka=%v", ka)
	} else if kb < 0 || kb > ka {
		err = fmt.Errorf("kb < 0 || kb > ka: kb=%v, ka=%v", kb, ka)
	} else if ab.Rows < ka+1 {
		err = fmt.Errorf("ab.Rows < ka+1: ab.Rows=%v, ka=%v", ab.Rows, ka)
	} else if bb.Rows < kb+1 {
		err = fmt.Errorf("bb.Rows < kb+1: bb.Rows=%v, kb=%v", bb.Rows, kb)
	} else if x.Rows < 1 || wantx && x.Rows < max(1, n) {
		err = fmt.Errorf("x.Rows < 1 || wantx && x.Rows < max(1, n): vect='%c', x.Rows=%v, n=%v", vect, x.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Dsbgst", err)
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}

	inca = ab.Rows * ka1

	//     Initialize X to the unit matrix, if needed
	if wantx {
		Dlaset(Full, n, n, zero, one, x)
	}

	//     Set M to the splitting point m. It must be the same value as is
	//     used in DPBSTF. The chosen value allows the arrays WORK and RWORK
	//     to be of dimension (N).
	m = (n + kb) / 2

	//     The routine works in two phases, corresponding to the two halves
	//     of the split Cholesky factorization of B as S**T*S where
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
	//     inv(S(i))**T*A*inv(S(i)). This creates a triangular bulge outside
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
	//     The cosines and sines of the rotations are stored in the array
	//     WORK. The cosines of the 1st set of rotations are stored in
	//     elements n+2:n+m-kb-1 and the sines of the 1st set in elements
	//     2:m-kb-1; the cosines of the 2nd set are stored in elements
	//     n+m-kb+1:2*n and the sines of the second set in elements m-kb+1:n.
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
	i = n + 1
label10:
	;
	if update {
		i = i - 1
		kbt = min(kb, i-1)
		i0 = i - 1
		i1 = min(n, i+ka)
		i2 = i - kbt + ka1
		if i < m+1 {
			update = false
			i = i + 1
			i0 = m
			if ka == 0 {
				goto label480
			}
			goto label10
		}
	} else {
		i = i + ka
		if i > n-1 {
			goto label480
		}
	}

	if upper {
		//        Transform A, working with the upper triangle
		if update {
			//           Form  inv(S(i))**T * A * inv(S(i))
			bii = bb.Get(kb1-1, i-1)
			for j = i; j <= i1; j++ {
				ab.Set(i-j+ka1-1, j-1, ab.Get(i-j+ka1-1, j-1)/bii)
			}
			for j = max(1, i-ka); j <= i; j++ {
				ab.Set(j-i+ka1-1, i-1, ab.Get(j-i+ka1-1, i-1)/bii)
			}
			for k = i - kbt; k <= i-1; k++ {
				for j = i - kbt; j <= k; j++ {
					ab.Set(j-k+ka1-1, k-1, ab.Get(j-k+ka1-1, k-1)-bb.Get(j-i+kb1-1, i-1)*ab.Get(k-i+ka1-1, i-1)-bb.Get(k-i+kb1-1, i-1)*ab.Get(j-i+ka1-1, i-1)+ab.Get(ka1-1, i-1)*bb.Get(j-i+kb1-1, i-1)*bb.Get(k-i+kb1-1, i-1))
				}
				for j = max(1, i-ka); j <= i-kbt-1; j++ {
					ab.Set(j-k+ka1-1, k-1, ab.Get(j-k+ka1-1, k-1)-bb.Get(k-i+kb1-1, i-1)*ab.Get(j-i+ka1-1, i-1))
				}
			}
			for j = i; j <= i1; j++ {
				for k = max(j-ka, i-kbt); k <= i-1; k++ {
					ab.Set(k-j+ka1-1, j-1, ab.Get(k-j+ka1-1, j-1)-bb.Get(k-i+kb1-1, i-1)*ab.Get(i-j+ka1-1, j-1))
				}
			}

			if wantx {
				//              post-multiply X by inv(S(i))
				x.Off(m, i-1).Vector().Scal(n-m, one/bii, 1)
				if kbt > 0 {
					err = x.Off(m, i-kbt-1).Ger(n-m, kbt, -one, x.Off(m, i-1).Vector(), 1, bb.Off(kb1-kbt-1, i-1).Vector(), 1)
				}
			}

			//           store a(i,i1) in RA1 for use in next loop over K
			ra1 = ab.Get(i-i1+ka1-1, i1-1)
		}

		//        Generate and apply vectors of rotations to chase all the
		//        existing bulges KA positions down toward the bottom of the
		//        band
		for k = 1; k <= kb-1; k++ {
			if update {
				//              Determine the rotations which would annihilate the bulge
				//              which has in theory just been created
				if i-k+ka < n && i-k > 1 {
					//                 generate rotation to annihilate a(i,i-k+ka+1)
					*work.GetPtr(n + i - k + ka - m - 1), *work.GetPtr(i - k + ka - m - 1), ra = Dlartg(ab.Get(k, i-k+ka-1), ra1)

					//                 create nonzero element a(i-k,i-k+ka+1) outside the
					//                 band and store it in WORK(i-k)
					t = -bb.Get(kb1-k-1, i-1) * ra1
					work.Set(i-k-1, work.Get(n+i-k+ka-m-1)*t-work.Get(i-k+ka-m-1)*ab.Get(0, i-k+ka-1))
					ab.Set(0, i-k+ka-1, work.Get(i-k+ka-m-1)*t+work.Get(n+i-k+ka-m-1)*ab.Get(0, i-k+ka-1))
					ra1 = ra
				}
			}
			j2 = i - k - 1 + max(1, k-i0+2)*ka1
			nr = (n - j2 + ka) / ka1
			j1 = j2 + (nr-1)*ka1
			if update {
				j2t = max(j2, i+2*ka-k+1)
			} else {
				j2t = j2
			}
			nrt = (n - j2t + ka) / ka1
			for j = j2t; j <= j1; j += ka1 {
				//              create nonzero element a(j-ka,j+1) outside the band
				//              and store it in WORK(j-m)
				work.Set(j-m-1, work.Get(j-m-1)*ab.Get(0, j))
				ab.Set(0, j, work.Get(n+j-m-1)*ab.Get(0, j))
			}

			//           generate rotations in 1st set to annihilate elements which
			//           have been created outside the band
			if nrt > 0 {
				Dlargv(nrt, ab.Off(0, j2t-1).Vector(), inca, work.Off(j2t-m-1), ka1, work.Off(n+j2t-m-1), ka1)
			}
			if nr > 0 {
				//              apply rotations in 1st set from the right
				for l = 1; l <= ka-1; l++ {
					Dlartv(nr, ab.Off(ka1-l-1, j2-1).Vector(), inca, ab.Off(ka-l-1, j2).Vector(), inca, work.Off(n+j2-m-1), work.Off(j2-m-1), ka1)
				}

				//              apply rotations in 1st set from both sides to diagonal
				//              blocks
				Dlar2v(nr, ab.Off(ka1-1, j2-1).Vector(), ab.Off(ka1-1, j2).Vector(), ab.Off(ka-1, j2).Vector(), inca, work.Off(n+j2-m-1), work.Off(j2-m-1), ka1)

			}

			//           start applying rotations in 1st set from the left
			for l = ka - 1; l >= kb-k+1; l-- {
				nrt = (n - j2 + l) / ka1
				if nrt > 0 {
					Dlartv(nrt, ab.Off(l-1, j2+ka1-l-1).Vector(), inca, ab.Off(l, j2+ka1-l-1).Vector(), inca, work.Off(n+j2-m-1), work.Off(j2-m-1), ka1)
				}
			}

			if wantx {
				//              post-multiply X by product of rotations in 1st set
				for j = j2; j <= j1; j += ka1 {
					x.Off(m, j).Vector().Rot(n-m, x.Off(m, j-1).Vector(), 1, 1, work.Get(n+j-m-1), work.Get(j-m-1))
				}
			}
		}

		if update {
			if i2 <= n && kbt > 0 {
				//              create nonzero element a(i-kbt,i-kbt+ka+1) outside the
				//              band and store it in WORK(i-kbt)
				work.Set(i-kbt-1, -bb.Get(kb1-kbt-1, i-1)*ra1)
			}
		}

		for k = kb; k >= 1; k-- {
			if update {
				j2 = i - k - 1 + max(2, k-i0+1)*ka1
			} else {
				j2 = i - k - 1 + max(1, k-i0+1)*ka1
			}

			//           finish applying rotations in 2nd set from the left
			for l = kb - k; l >= 1; l-- {
				nrt = (n - j2 + ka + l) / ka1
				if nrt > 0 {
					Dlartv(nrt, ab.Off(l-1, j2-l).Vector(), inca, ab.Off(l, j2-l).Vector(), inca, work.Off(n+j2-ka-1), work.Off(j2-ka-1), ka1)
				}
			}
			nr = (n - j2 + ka) / ka1
			j1 = j2 + (nr-1)*ka1
			for j = j1; j >= j2; j -= ka1 {
				work.Set(j-1, work.Get(j-ka-1))
				work.Set(n+j-1, work.Get(n+j-ka-1))
			}
			for j = j2; j <= j1; j += ka1 {
				//              create nonzero element a(j-ka,j+1) outside the band
				//              and store it in WORK(j)
				work.Set(j-1, work.Get(j-1)*ab.Get(0, j))
				ab.Set(0, j, work.Get(n+j-1)*ab.Get(0, j))
			}
			if update {
				if i-k < n-ka && k <= kbt {
					work.Set(i-k+ka-1, work.Get(i-k-1))
				}
			}
		}

		for k = kb; k >= 1; k-- {
			j2 = i - k - 1 + max(1, k-i0+1)*ka1
			nr = (n - j2 + ka) / ka1
			j1 = j2 + (nr-1)*ka1
			if nr > 0 {
				//              generate rotations in 2nd set to annihilate elements
				//              which have been created outside the band
				Dlargv(nr, ab.Off(0, j2-1).Vector(), inca, work.Off(j2-1), ka1, work.Off(n+j2-1), ka1)

				//              apply rotations in 2nd set from the right
				for l = 1; l <= ka-1; l++ {
					Dlartv(nr, ab.Off(ka1-l-1, j2-1).Vector(), inca, ab.Off(ka-l-1, j2).Vector(), inca, work.Off(n+j2-1), work.Off(j2-1), ka1)
				}

				//              apply rotations in 2nd set from both sides to diagonal
				//              blocks
				Dlar2v(nr, ab.Off(ka1-1, j2-1).Vector(), ab.Off(ka1-1, j2).Vector(), ab.Off(ka-1, j2).Vector(), inca, work.Off(n+j2-1), work.Off(j2-1), ka1)

			}

			//           start applying rotations in 2nd set from the left
			for l = ka - 1; l >= kb-k+1; l-- {
				nrt = (n - j2 + l) / ka1
				if nrt > 0 {
					Dlartv(nrt, ab.Off(l-1, j2+ka1-l-1).Vector(), inca, ab.Off(l, j2+ka1-l-1).Vector(), inca, work.Off(n+j2-1), work.Off(j2-1), ka1)
				}
			}

			if wantx {
				//              post-multiply X by product of rotations in 2nd set
				for j = j2; j <= j1; j += ka1 {
					x.Off(m, j).Vector().Rot(n-m, x.Off(m, j-1).Vector(), 1, 1, work.Get(n+j-1), work.Get(j-1))
				}
			}
		}

		for k = 1; k <= kb-1; k++ {
			j2 = i - k - 1 + max(1, k-i0+2)*ka1

			//           finish applying rotations in 1st set from the left
			for l = kb - k; l >= 1; l-- {
				nrt = (n - j2 + l) / ka1
				if nrt > 0 {
					Dlartv(nrt, ab.Off(l-1, j2+ka1-l-1).Vector(), inca, ab.Off(l, j2+ka1-l-1).Vector(), inca, work.Off(n+j2-m-1), work.Off(j2-m-1), ka1)
				}
			}
		}

		if kb > 1 {
			for j = n - 1; j >= i-kb+2*ka+1; j-- {
				work.Set(n+j-m-1, work.Get(n+j-ka-m-1))
				work.Set(j-m-1, work.Get(j-ka-m-1))
			}
		}

	} else {
		//        Transform A, working with the lower triangle
		if update {
			//           Form  inv(S(i))**T * A * inv(S(i))
			bii = bb.Get(0, i-1)
			for j = i; j <= i1; j++ {
				ab.Set(j-i, i-1, ab.Get(j-i, i-1)/bii)
			}
			for j = max(1, i-ka); j <= i; j++ {
				ab.Set(i-j, j-1, ab.Get(i-j, j-1)/bii)
			}
			for k = i - kbt; k <= i-1; k++ {
				for j = i - kbt; j <= k; j++ {
					ab.Set(k-j, j-1, ab.Get(k-j, j-1)-bb.Get(i-j, j-1)*ab.Get(i-k, k-1)-bb.Get(i-k, k-1)*ab.Get(i-j, j-1)+ab.Get(0, i-1)*bb.Get(i-j, j-1)*bb.Get(i-k, k-1))
				}
				for j = max(1, i-ka); j <= i-kbt-1; j++ {
					ab.Set(k-j, j-1, ab.Get(k-j, j-1)-bb.Get(i-k, k-1)*ab.Get(i-j, j-1))
				}
			}
			for j = i; j <= i1; j++ {
				for k = max(j-ka, i-kbt); k <= i-1; k++ {
					ab.Set(j-k, k-1, ab.Get(j-k, k-1)-bb.Get(i-k, k-1)*ab.Get(j-i, i-1))
				}
			}

			if wantx {
				//              post-multiply X by inv(S(i))
				x.Off(m, i-1).Vector().Scal(n-m, one/bii, 1)
				if kbt > 0 {
					err = x.Off(m, i-kbt-1).Ger(n-m, kbt, -one, x.Off(m, i-1).Vector(), 1, bb.Off(kbt, i-kbt-1).Vector(), bb.Rows-1)
				}
			}

			//           store a(i1,i) in RA1 for use in next loop over K
			ra1 = ab.Get(i1-i, i-1)
		}

		//        Generate and apply vectors of rotations to chase all the
		//        existing bulges KA positions down toward the bottom of the
		//        band
		for k = 1; k <= kb-1; k++ {
			if update {
				//              Determine the rotations which would annihilate the bulge
				//              which has in theory just been created
				if i-k+ka < n && i-k > 1 {
					//                 generate rotation to annihilate a(i-k+ka+1,i)
					*work.GetPtr(n + i - k + ka - m - 1), *work.GetPtr(i - k + ka - m - 1), ra = Dlartg(ab.Get(ka1-k-1, i-1), ra1)

					//                 create nonzero element a(i-k+ka+1,i-k) outside the
					//                 band and store it in WORK(i-k)
					t = -bb.Get(k, i-k-1) * ra1
					work.Set(i-k-1, work.Get(n+i-k+ka-m-1)*t-work.Get(i-k+ka-m-1)*ab.Get(ka1-1, i-k-1))
					ab.Set(ka1-1, i-k-1, work.Get(i-k+ka-m-1)*t+work.Get(n+i-k+ka-m-1)*ab.Get(ka1-1, i-k-1))
					ra1 = ra
				}
			}
			j2 = i - k - 1 + max(1, k-i0+2)*ka1
			nr = (n - j2 + ka) / ka1
			j1 = j2 + (nr-1)*ka1
			if update {
				j2t = max(j2, i+2*ka-k+1)
			} else {
				j2t = j2
			}
			nrt = (n - j2t + ka) / ka1
			for j = j2t; j <= j1; j += ka1 {
				//              create nonzero element a(j+1,j-ka) outside the band
				//              and store it in WORK(j-m)
				work.Set(j-m-1, work.Get(j-m-1)*ab.Get(ka1-1, j-ka))
				ab.Set(ka1-1, j-ka, work.Get(n+j-m-1)*ab.Get(ka1-1, j-ka))
			}

			//           generate rotations in 1st set to annihilate elements which
			//           have been created outside the band
			if nrt > 0 {
				Dlargv(nrt, ab.Off(ka1-1, j2t-ka-1).Vector(), inca, work.Off(j2t-m-1), ka1, work.Off(n+j2t-m-1), ka1)
			}
			if nr > 0 {
				//              apply rotations in 1st set from the left
				for l = 1; l <= ka-1; l++ {
					Dlartv(nr, ab.Off(l, j2-l-1).Vector(), inca, ab.Off(l+2-1, j2-l-1).Vector(), inca, work.Off(n+j2-m-1), work.Off(j2-m-1), ka1)
				}

				//              apply rotations in 1st set from both sides to diagonal
				//              blocks
				Dlar2v(nr, ab.Off(0, j2-1).Vector(), ab.Off(0, j2).Vector(), ab.Off(1, j2-1).Vector(), inca, work.Off(n+j2-m-1), work.Off(j2-m-1), ka1)
				//
			}

			//           start applying rotations in 1st set from the right
			for l = ka - 1; l >= kb-k+1; l-- {
				nrt = (n - j2 + l) / ka1
				if nrt > 0 {
					Dlartv(nrt, ab.Off(ka1-l, j2-1).Vector(), inca, ab.Off(ka1-l-1, j2).Vector(), inca, work.Off(n+j2-m-1), work.Off(j2-m-1), ka1)
				}
			}

			if wantx {
				//              post-multiply X by product of rotations in 1st set
				for j = j2; j <= j1; j += ka1 {
					x.Off(m, j).Vector().Rot(n-m, x.Off(m, j-1).Vector(), 1, 1, work.Get(n+j-m-1), work.Get(j-m-1))
				}
			}
		}

		if update {
			if i2 <= n && kbt > 0 {
				//              create nonzero element a(i-kbt+ka+1,i-kbt) outside the
				//              band and store it in WORK(i-kbt)
				work.Set(i-kbt-1, -bb.Get(kbt, i-kbt-1)*ra1)
			}
		}

		for k = kb; k >= 1; k-- {
			if update {
				j2 = i - k - 1 + max(2, k-i0+1)*ka1
			} else {
				j2 = i - k - 1 + max(1, k-i0+1)*ka1
			}

			//           finish applying rotations in 2nd set from the right
			for l = kb - k; l >= 1; l-- {
				nrt = (n - j2 + ka + l) / ka1
				if nrt > 0 {
					Dlartv(nrt, ab.Off(ka1-l, j2-ka-1).Vector(), inca, ab.Off(ka1-l-1, j2-ka).Vector(), inca, work.Off(n+j2-ka-1), work.Off(j2-ka-1), ka1)
				}
			}
			nr = (n - j2 + ka) / ka1
			j1 = j2 + (nr-1)*ka1
			for j = j1; j >= j2; j -= ka1 {
				work.Set(j-1, work.Get(j-ka-1))
				work.Set(n+j-1, work.Get(n+j-ka-1))
			}
			for j = j2; j <= j1; j += ka1 {
				//              create nonzero element a(j+1,j-ka) outside the band
				//              and store it in WORK(j)
				work.Set(j-1, work.Get(j-1)*ab.Get(ka1-1, j-ka))
				ab.Set(ka1-1, j-ka, work.Get(n+j-1)*ab.Get(ka1-1, j-ka))
			}
			if update {
				if i-k < n-ka && k <= kbt {
					work.Set(i-k+ka-1, work.Get(i-k-1))
				}
			}
		}

		for k = kb; k >= 1; k-- {
			j2 = i - k - 1 + max(1, k-i0+1)*ka1
			nr = (n - j2 + ka) / ka1
			j1 = j2 + (nr-1)*ka1
			if nr > 0 {
				//              generate rotations in 2nd set to annihilate elements
				//              which have been created outside the band
				Dlargv(nr, ab.Off(ka1-1, j2-ka-1).Vector(), inca, work.Off(j2-1), ka1, work.Off(n+j2-1), ka1)

				//              apply rotations in 2nd set from the left
				for l = 1; l <= ka-1; l++ {
					Dlartv(nr, ab.Off(l, j2-l-1).Vector(), inca, ab.Off(l+2-1, j2-l-1).Vector(), inca, work.Off(n+j2-1), work.Off(j2-1), ka1)
				}

				//              apply rotations in 2nd set from both sides to diagonal
				//              blocks
				Dlar2v(nr, ab.Off(0, j2-1).Vector(), ab.Off(0, j2).Vector(), ab.Off(1, j2-1).Vector(), inca, work.Off(n+j2-1), work.Off(j2-1), ka1)

			}

			//           start applying rotations in 2nd set from the right
			for l = ka - 1; l >= kb-k+1; l-- {
				nrt = (n - j2 + l) / ka1
				if nrt > 0 {
					Dlartv(nrt, ab.Off(ka1-l, j2-1).Vector(), inca, ab.Off(ka1-l-1, j2).Vector(), inca, work.Off(n+j2-1), work.Off(j2-1), ka1)
				}
			}

			if wantx {
				//              post-multiply X by product of rotations in 2nd set
				for j = j2; j <= j1; j += ka1 {
					x.Off(m, j).Vector().Rot(n-m, x.Off(m, j-1).Vector(), 1, 1, work.Get(n+j-1), work.Get(j-1))
				}
			}
		}

		for k = 1; k <= kb-1; k++ {
			j2 = i - k - 1 + max(1, k-i0+2)*ka1

			//           finish applying rotations in 1st set from the right
			for l = kb - k; l >= 1; l-- {
				nrt = (n - j2 + l) / ka1
				if nrt > 0 {
					Dlartv(nrt, ab.Off(ka1-l, j2-1).Vector(), inca, ab.Off(ka1-l-1, j2).Vector(), inca, work.Off(n+j2-m-1), work.Off(j2-m-1), ka1)
				}
			}
		}

		if kb > 1 {
			for j = n - 1; j >= i-kb+2*ka+1; j-- {
				work.Set(n+j-m-1, work.Get(n+j-ka-m-1))
				work.Set(j-m-1, work.Get(j-ka-m-1))
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
		kbt = min(kb, m-i)
		i0 = i + 1
		i1 = max(1, i-ka)
		i2 = i + kbt - ka1
		if i > m {
			update = false
			i = i - 1
			i0 = m + 1
			if ka == 0 {
				return
			}
			goto label490
		}
	} else {
		i = i - ka
		if i < 2 {
			return
		}
	}

	if i < m-kbt {
		nx = m
	} else {
		nx = n
	}

	if upper {
		//        Transform A, working with the upper triangle
		if update {
			//           Form  inv(S(i))**T * A * inv(S(i))
			bii = bb.Get(kb1-1, i-1)
			for j = i1; j <= i; j++ {
				ab.Set(j-i+ka1-1, i-1, ab.Get(j-i+ka1-1, i-1)/bii)
			}
			for j = i; j <= min(n, i+ka); j++ {
				ab.Set(i-j+ka1-1, j-1, ab.Get(i-j+ka1-1, j-1)/bii)
			}
			for k = i + 1; k <= i+kbt; k++ {
				for j = k; j <= i+kbt; j++ {
					ab.Set(k-j+ka1-1, j-1, ab.Get(k-j+ka1-1, j-1)-bb.Get(i-j+kb1-1, j-1)*ab.Get(i-k+ka1-1, k-1)-bb.Get(i-k+kb1-1, k-1)*ab.Get(i-j+ka1-1, j-1)+ab.Get(ka1-1, i-1)*bb.Get(i-j+kb1-1, j-1)*bb.Get(i-k+kb1-1, k-1))
				}
				for j = i + kbt + 1; j <= min(n, i+ka); j++ {
					ab.Set(k-j+ka1-1, j-1, ab.Get(k-j+ka1-1, j-1)-bb.Get(i-k+kb1-1, k-1)*ab.Get(i-j+ka1-1, j-1))
				}
			}
			for j = i1; j <= i; j++ {
				for k = i + 1; k <= min(j+ka, i+kbt); k++ {
					ab.Set(j-k+ka1-1, k-1, ab.Get(j-k+ka1-1, k-1)-bb.Get(i-k+kb1-1, k-1)*ab.Get(j-i+ka1-1, i-1))
				}
			}

			if wantx {
				//              post-multiply X by inv(S(i))
				x.Off(0, i-1).Vector().Scal(nx, one/bii, 1)
				if kbt > 0 {
					err = x.Off(0, i).Ger(nx, kbt, -one, x.Off(0, i-1).Vector(), 1, bb.Off(kb-1, i).Vector(), bb.Rows-1)
				}
			}

			//           store a(i1,i) in RA1 for use in next loop over K
			ra1 = ab.Get(i1-i+ka1-1, i-1)
		}

		//        Generate and apply vectors of rotations to chase all the
		//        existing bulges KA positions up toward the top of the band
		for k = 1; k <= kb-1; k++ {
			if update {
				//              Determine the rotations which would annihilate the bulge
				//              which has in theory just been created
				if i+k-ka1 > 0 && i+k < m {
					//                 generate rotation to annihilate a(i+k-ka-1,i)
					*work.GetPtr(n + i + k - ka - 1), *work.GetPtr(i + k - ka - 1), ra = Dlartg(ab.Get(k, i-1), ra1)

					//                 create nonzero element a(i+k-ka-1,i+k) outside the
					//                 band and store it in WORK(m-kb+i+k)
					t = -bb.Get(kb1-k-1, i+k-1) * ra1
					work.Set(m-kb+i+k-1, work.Get(n+i+k-ka-1)*t-work.Get(i+k-ka-1)*ab.Get(0, i+k-1))
					ab.Set(0, i+k-1, work.Get(i+k-ka-1)*t+work.Get(n+i+k-ka-1)*ab.Get(0, i+k-1))
					ra1 = ra
				}
			}
			j2 = i + k + 1 - max(1, k+i0-m+1)*ka1
			nr = (j2 + ka - 1) / ka1
			j1 = j2 - (nr-1)*ka1
			if update {
				j2t = min(j2, i-2*ka+k-1)
			} else {
				j2t = j2
			}
			nrt = (j2t + ka - 1) / ka1
			for j = j1; j <= j2t; j += ka1 {
				//              create nonzero element a(j-1,j+ka) outside the band
				//              and store it in WORK(j)
				work.Set(j-1, work.Get(j-1)*ab.Get(0, j+ka-1-1))
				ab.Set(0, j+ka-1-1, work.Get(n+j-1)*ab.Get(0, j+ka-1-1))
			}

			//           generate rotations in 1st set to annihilate elements which
			//           have been created outside the band
			if nrt > 0 {
				Dlargv(nrt, ab.Off(0, j1+ka-1).Vector(), inca, work.Off(j1-1), ka1, work.Off(n+j1-1), ka1)
			}
			if nr > 0 {
				//              apply rotations in 1st set from the left
				for l = 1; l <= ka-1; l++ {
					Dlartv(nr, ab.Off(ka1-l-1, j1+l-1).Vector(), inca, ab.Off(ka-l-1, j1+l-1).Vector(), inca, work.Off(n+j1-1), work.Off(j1-1), ka1)
				}

				//              apply rotations in 1st set from both sides to diagonal
				//              blocks
				Dlar2v(nr, ab.Off(ka1-1, j1-1).Vector(), ab.Off(ka1-1, j1-1-1).Vector(), ab.Off(ka-1, j1-1).Vector(), inca, work.Off(n+j1-1), work.Off(j1-1), ka1)

			}

			//           start applying rotations in 1st set from the right
			for l = ka - 1; l >= kb-k+1; l-- {
				nrt = (j2 + l - 1) / ka1
				j1t = j2 - (nrt-1)*ka1
				if nrt > 0 {
					Dlartv(nrt, ab.Off(l-1, j1t-1).Vector(), inca, ab.Off(l, j1t-1-1).Vector(), inca, work.Off(n+j1t-1), work.Off(j1t-1), ka1)
				}
			}

			if wantx {
				//              post-multiply X by product of rotations in 1st set
				for j = j1; j <= j2; j += ka1 {
					x.Off(0, j-1-1).Vector().Rot(nx, x.Off(0, j-1).Vector(), 1, 1, work.Get(n+j-1), work.Get(j-1))
				}
			}
		}

		if update {
			if i2 > 0 && kbt > 0 {
				//              create nonzero element a(i+kbt-ka-1,i+kbt) outside the
				//              band and store it in WORK(m-kb+i+kbt)
				work.Set(m-kb+i+kbt-1, -bb.Get(kb1-kbt-1, i+kbt-1)*ra1)
			}
		}

		for k = kb; k >= 1; k-- {
			if update {
				j2 = i + k + 1 - max(2, k+i0-m)*ka1
			} else {
				j2 = i + k + 1 - max(1, k+i0-m)*ka1
			}

			//           finish applying rotations in 2nd set from the right
			for l = kb - k; l >= 1; l-- {
				nrt = (j2 + ka + l - 1) / ka1
				j1t = j2 - (nrt-1)*ka1
				if nrt > 0 {
					Dlartv(nrt, ab.Off(l-1, j1t+ka-1).Vector(), inca, ab.Off(l, j1t+ka-1-1).Vector(), inca, work.Off(n+m-kb+j1t+ka-1), work.Off(m-kb+j1t+ka-1), ka1)
				}
			}
			nr = (j2 + ka - 1) / ka1
			j1 = j2 - (nr-1)*ka1
			for j = j1; j <= j2; j += ka1 {
				work.Set(m-kb+j-1, work.Get(m-kb+j+ka-1))
				work.Set(n+m-kb+j-1, work.Get(n+m-kb+j+ka-1))
			}
			for j = j1; j <= j2; j += ka1 {
				//              create nonzero element a(j-1,j+ka) outside the band
				//              and store it in WORK(m-kb+j)
				work.Set(m-kb+j-1, work.Get(m-kb+j-1)*ab.Get(0, j+ka-1-1))
				ab.Set(0, j+ka-1-1, work.Get(n+m-kb+j-1)*ab.Get(0, j+ka-1-1))
			}
			if update {
				if i+k > ka1 && k <= kbt {
					work.Set(m-kb+i+k-ka-1, work.Get(m-kb+i+k-1))
				}
			}
		}

		for k = kb; k >= 1; k-- {
			j2 = i + k + 1 - max(1, k+i0-m)*ka1
			nr = (j2 + ka - 1) / ka1
			j1 = j2 - (nr-1)*ka1
			if nr > 0 {
				//              generate rotations in 2nd set to annihilate elements
				//              which have been created outside the band
				Dlargv(nr, ab.Off(0, j1+ka-1).Vector(), inca, work.Off(m-kb+j1-1), ka1, work.Off(n+m-kb+j1-1), ka1)

				//              apply rotations in 2nd set from the left
				for l = 1; l <= ka-1; l++ {
					Dlartv(nr, ab.Off(ka1-l-1, j1+l-1).Vector(), inca, ab.Off(ka-l-1, j1+l-1).Vector(), inca, work.Off(n+m-kb+j1-1), work.Off(m-kb+j1-1), ka1)
				}

				//              apply rotations in 2nd set from both sides to diagonal
				//              blocks
				Dlar2v(nr, ab.Off(ka1-1, j1-1).Vector(), ab.Off(ka1-1, j1-1-1).Vector(), ab.Off(ka-1, j1-1).Vector(), inca, work.Off(n+m-kb+j1-1), work.Off(m-kb+j1-1), ka1)

			}

			//           start applying rotations in 2nd set from the right
			for l = ka - 1; l >= kb-k+1; l-- {
				nrt = (j2 + l - 1) / ka1
				j1t = j2 - (nrt-1)*ka1
				if nrt > 0 {
					Dlartv(nrt, ab.Off(l-1, j1t-1).Vector(), inca, ab.Off(l, j1t-1-1).Vector(), inca, work.Off(n+m-kb+j1t-1), work.Off(m-kb+j1t-1), ka1)
				}
			}

			if wantx {
				//              post-multiply X by product of rotations in 2nd set
				for j = j1; j <= j2; j += ka1 {
					x.Off(0, j-1-1).Vector().Rot(nx, x.Off(0, j-1).Vector(), 1, 1, work.Get(n+m-kb+j-1), work.Get(m-kb+j-1))
				}
			}
		}

		for k = 1; k <= kb-1; k++ {
			j2 = i + k + 1 - max(1, k+i0-m+1)*ka1

			//           finish applying rotations in 1st set from the right
			for l = kb - k; l >= 1; l-- {
				nrt = (j2 + l - 1) / ka1
				j1t = j2 - (nrt-1)*ka1
				if nrt > 0 {
					Dlartv(nrt, ab.Off(l-1, j1t-1).Vector(), inca, ab.Off(l, j1t-1-1).Vector(), inca, work.Off(n+j1t-1), work.Off(j1t-1), ka1)
				}
			}
		}

		if kb > 1 {
			for j = 2; j <= min(i+kb, m)-2*ka-1; j++ {
				work.Set(n+j-1, work.Get(n+j+ka-1))
				work.Set(j-1, work.Get(j+ka-1))
			}
		}

	} else {
		//        Transform A, working with the lower triangle
		if update {
			//           Form  inv(S(i))**T * A * inv(S(i))
			bii = bb.Get(0, i-1)
			for j = i1; j <= i; j++ {
				ab.Set(i-j, j-1, ab.Get(i-j, j-1)/bii)
			}
			for j = i; j <= min(n, i+ka); j++ {
				ab.Set(j-i, i-1, ab.Get(j-i, i-1)/bii)
			}
			for k = i + 1; k <= i+kbt; k++ {
				for j = k; j <= i+kbt; j++ {
					ab.Set(j-k, k-1, ab.Get(j-k, k-1)-bb.Get(j-i, i-1)*ab.Get(k-i, i-1)-bb.Get(k-i, i-1)*ab.Get(j-i, i-1)+ab.Get(0, i-1)*bb.Get(j-i, i-1)*bb.Get(k-i, i-1))
				}
				for j = i + kbt + 1; j <= min(n, i+ka); j++ {
					ab.Set(j-k, k-1, ab.Get(j-k, k-1)-bb.Get(k-i, i-1)*ab.Get(j-i, i-1))
				}
			}
			for j = i1; j <= i; j++ {
				for k = i + 1; k <= min(j+ka, i+kbt); k++ {
					ab.Set(k-j, j-1, ab.Get(k-j, j-1)-bb.Get(k-i, i-1)*ab.Get(i-j, j-1))
				}
			}

			if wantx {
				//              post-multiply X by inv(S(i))
				x.Off(0, i-1).Vector().Scal(nx, one/bii, 1)
				if kbt > 0 {
					err = x.Off(0, i).Ger(nx, kbt, -one, x.Off(0, i-1).Vector(), 1, bb.Off(1, i-1).Vector(), 1)
				}
			}

			//           store a(i,i1) in RA1 for use in next loop over K
			ra1 = ab.Get(i-i1, i1-1)
		}

		//        Generate and apply vectors of rotations to chase all the
		//        existing bulges KA positions up toward the top of the band
		for k = 1; k <= kb-1; k++ {
			if update {
				//              Determine the rotations which would annihilate the bulge
				//              which has in theory just been created
				if i+k-ka1 > 0 && i+k < m {
					//                 generate rotation to annihilate a(i,i+k-ka-1)
					*work.GetPtr(n + i + k - ka - 1), *work.GetPtr(i + k - ka - 1), ra = Dlartg(ab.Get(ka1-k-1, i+k-ka-1), ra1)

					//                 create nonzero element a(i+k,i+k-ka-1) outside the
					//                 band and store it in WORK(m-kb+i+k)
					t = -bb.Get(k, i-1) * ra1
					work.Set(m-kb+i+k-1, work.Get(n+i+k-ka-1)*t-work.Get(i+k-ka-1)*ab.Get(ka1-1, i+k-ka-1))
					ab.Set(ka1-1, i+k-ka-1, work.Get(i+k-ka-1)*t+work.Get(n+i+k-ka-1)*ab.Get(ka1-1, i+k-ka-1))
					ra1 = ra
				}
			}
			j2 = i + k + 1 - max(1, k+i0-m+1)*ka1
			nr = (j2 + ka - 1) / ka1
			j1 = j2 - (nr-1)*ka1
			if update {
				j2t = min(j2, i-2*ka+k-1)
			} else {
				j2t = j2
			}
			nrt = (j2t + ka - 1) / ka1
			for j = j1; j <= j2t; j += ka1 {
				//              create nonzero element a(j+ka,j-1) outside the band
				//              and store it in WORK(j)
				work.Set(j-1, work.Get(j-1)*ab.Get(ka1-1, j-1-1))
				ab.Set(ka1-1, j-1-1, work.Get(n+j-1)*ab.Get(ka1-1, j-1-1))
			}

			//           generate rotations in 1st set to annihilate elements which
			//           have been created outside the band
			if nrt > 0 {
				Dlargv(nrt, ab.Off(ka1-1, j1-1).Vector(), inca, work.Off(j1-1), ka1, work.Off(n+j1-1), ka1)
			}
			if nr > 0 {
				//              apply rotations in 1st set from the right
				for l = 1; l <= ka-1; l++ {
					Dlartv(nr, ab.Off(l, j1-1).Vector(), inca, ab.Off(l+2-1, j1-1-1).Vector(), inca, work.Off(n+j1-1), work.Off(j1-1), ka1)
				}

				//              apply rotations in 1st set from both sides to diagonal
				//              blocks
				Dlar2v(nr, ab.Off(0, j1-1).Vector(), ab.Off(0, j1-1-1).Vector(), ab.Off(1, j1-1-1).Vector(), inca, work.Off(n+j1-1), work.Off(j1-1), ka1)

			}

			//           start applying rotations in 1st set from the left
			for l = ka - 1; l >= kb-k+1; l-- {
				nrt = (j2 + l - 1) / ka1
				j1t = j2 - (nrt-1)*ka1
				if nrt > 0 {
					Dlartv(nrt, ab.Off(ka1-l, j1t-ka1+l-1).Vector(), inca, ab.Off(ka1-l-1, j1t-ka1+l-1).Vector(), inca, work.Off(n+j1t-1), work.Off(j1t-1), ka1)
				}
			}

			if wantx {
				//              post-multiply X by product of rotations in 1st set
				for j = j1; j <= j2; j += ka1 {
					x.Off(0, j-1-1).Vector().Rot(nx, x.Off(0, j-1).Vector(), 1, 1, work.Get(n+j-1), work.Get(j-1))
				}
			}
		}

		if update {
			if i2 > 0 && kbt > 0 {
				//              create nonzero element a(i+kbt,i+kbt-ka-1) outside the
				//              band and store it in WORK(m-kb+i+kbt)
				work.Set(m-kb+i+kbt-1, -bb.Get(kbt, i-1)*ra1)
			}
		}

		for k = kb; k >= 1; k-- {
			if update {
				j2 = i + k + 1 - max(2, k+i0-m)*ka1
			} else {
				j2 = i + k + 1 - max(1, k+i0-m)*ka1
			}

			//           finish applying rotations in 2nd set from the left
			for l = kb - k; l >= 1; l-- {
				nrt = (j2 + ka + l - 1) / ka1
				j1t = j2 - (nrt-1)*ka1
				if nrt > 0 {
					Dlartv(nrt, ab.Off(ka1-l, j1t+l-1-1).Vector(), inca, ab.Off(ka1-l-1, j1t+l-1-1).Vector(), inca, work.Off(n+m-kb+j1t+ka-1), work.Off(m-kb+j1t+ka-1), ka1)
				}
			}
			nr = (j2 + ka - 1) / ka1
			j1 = j2 - (nr-1)*ka1
			for j = j1; j <= j2; j += ka1 {
				work.Set(m-kb+j-1, work.Get(m-kb+j+ka-1))
				work.Set(n+m-kb+j-1, work.Get(n+m-kb+j+ka-1))
			}
			for j = j1; j <= j2; j += ka1 {
				//              create nonzero element a(j+ka,j-1) outside the band
				//              and store it in WORK(m-kb+j)
				work.Set(m-kb+j-1, work.Get(m-kb+j-1)*ab.Get(ka1-1, j-1-1))
				ab.Set(ka1-1, j-1-1, work.Get(n+m-kb+j-1)*ab.Get(ka1-1, j-1-1))
			}
			if update {
				if i+k > ka1 && k <= kbt {
					work.Set(m-kb+i+k-ka-1, work.Get(m-kb+i+k-1))
				}
			}
		}

		for k = kb; k >= 1; k-- {
			j2 = i + k + 1 - max(1, k+i0-m)*ka1
			nr = (j2 + ka - 1) / ka1
			j1 = j2 - (nr-1)*ka1
			if nr > 0 {
				//              generate rotations in 2nd set to annihilate elements
				//              which have been created outside the band
				Dlargv(nr, ab.Off(ka1-1, j1-1).Vector(), inca, work.Off(m-kb+j1-1), ka1, work.Off(n+m-kb+j1-1), ka1)

				//              apply rotations in 2nd set from the right
				for l = 1; l <= ka-1; l++ {
					Dlartv(nr, ab.Off(l, j1-1).Vector(), inca, ab.Off(l+2-1, j1-1-1).Vector(), inca, work.Off(n+m-kb+j1-1), work.Off(m-kb+j1-1), ka1)
				}

				//              apply rotations in 2nd set from both sides to diagonal
				//              blocks
				Dlar2v(nr, ab.Off(0, j1-1).Vector(), ab.Off(0, j1-1-1).Vector(), ab.Off(1, j1-1-1).Vector(), inca, work.Off(n+m-kb+j1-1), work.Off(m-kb+j1-1), ka1)

			}

			//           start applying rotations in 2nd set from the left
			for l = ka - 1; l >= kb-k+1; l-- {
				nrt = (j2 + l - 1) / ka1
				j1t = j2 - (nrt-1)*ka1
				if nrt > 0 {
					Dlartv(nrt, ab.Off(ka1-l, j1t-ka1+l-1).Vector(), inca, ab.Off(ka1-l-1, j1t-ka1+l-1).Vector(), inca, work.Off(n+m-kb+j1t-1), work.Off(m-kb+j1t-1), ka1)
				}
			}

			if wantx {
				//              post-multiply X by product of rotations in 2nd set
				for j = j1; j <= j2; j += ka1 {
					x.Off(0, j-1-1).Vector().Rot(nx, x.Off(0, j-1).Vector(), 1, 1, work.Get(n+m-kb+j-1), work.Get(m-kb+j-1))
				}
			}
		}

		for k = 1; k <= kb-1; k++ {
			j2 = i + k + 1 - max(1, k+i0-m+1)*ka1

			//           finish applying rotations in 1st set from the left
			for l = kb - k; l >= 1; l-- {
				nrt = (j2 + l - 1) / ka1
				j1t = j2 - (nrt-1)*ka1
				if nrt > 0 {
					Dlartv(nrt, ab.Off(ka1-l, j1t-ka1+l-1).Vector(), inca, ab.Off(ka1-l-1, j1t-ka1+l-1).Vector(), inca, work.Off(n+j1t-1), work.Off(j1t-1), ka1)
				}
			}
		}

		if kb > 1 {
			for j = 2; j <= min(i+kb, m)-2*ka-1; j++ {
				work.Set(n+j-1, work.Get(n+j+ka-1))
				work.Set(j-1, work.Get(j+ka-1))
			}
		}

	}

	goto label490
}
