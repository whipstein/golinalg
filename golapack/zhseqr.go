package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zhseqr computes the eigenvalues of a Hessenberg matrix H
//    and, optionally, the matrices T and Z from the Schur decomposition
//    H = Z T Z**H, where T is an upper triangular matrix (the
//    Schur form), and Z is the unitary matrix of Schur vectors.
//
//    Optionally Z may be postmultiplied into an input unitary
//    matrix Q so that this routine can give the Schur factorization
//    of a matrix A which has been reduced to the Hessenberg form H
//    by the unitary matrix Q:  A = Q*H*Q**H = (QZ)*T*(QZ)**H.
func Zhseqr(job, compz byte, n, ilo, ihi *int, h *mat.CMatrix, ldh *int, w *mat.CVector, z *mat.CMatrix, ldz *int, work *mat.CVector, lwork, info *int) {
	var initz, lquery, wantt, wantz bool
	var one, zero complex128
	var rzero float64
	var kbot, nl, nmin, ntiny int
	workl := cvf(49)
	hl := cmf(49, 49, opts)

	ntiny = 11

	//     ==== NL allocates some local workspace to help small matrices
	//     .    through a rare ZLAHQR failure.  NL > NTINY = 11 is
	//     .    required and NL <= NMIN = ILAENV(ISPEC=12,...) is recom-
	//     .    mended.  (The default value of NMIN is 75.)  Using NL = 49
	//     .    allows up to six simultaneous shifts and a 16-by-16
	//     .    deflation window.  ====
	nl = 49
	zero = (0.0 + 0.0*1i)
	one = (1.0 + 0.0*1i)
	rzero = 0.0

	//     ==== Decode and check the input parameters. ====
	wantt = job == 'S'
	initz = compz == 'I'
	wantz = initz || compz == 'V'
	work.Set(0, complex(float64(max(1, *n)), rzero))
	lquery = (*lwork) == -1

	(*info) = 0
	if job != 'E' && !wantt {
		(*info) = -1
	} else if compz != 'N' && !wantz {
		(*info) = -2
	} else if (*n) < 0 {
		(*info) = -3
	} else if (*ilo) < 1 || (*ilo) > max(1, *n) {
		(*info) = -4
	} else if (*ihi) < min(*ilo, *n) || (*ihi) > (*n) {
		(*info) = -5
	} else if (*ldh) < max(1, *n) {
		(*info) = -7
	} else if (*ldz) < 1 || (wantz && (*ldz) < max(1, *n)) {
		(*info) = -10
	} else if (*lwork) < max(1, *n) && !lquery {
		(*info) = -12
	}

	if (*info) != 0 {
		//        ==== Quick return in case of invalid argument. ====
		gltest.Xerbla([]byte("ZHSEQR"), -(*info))
		return

	} else if (*n) == 0 {
		//        ==== Quick return in case N = 0; nothing to do. ====
		return

	} else if lquery {
		//        ==== Quick return in case of a workspace query ====
		Zlaqr0(wantt, wantz, n, ilo, ihi, h, ldh, w, ilo, ihi, z, ldz, work, lwork, info)
		//        ==== Ensure reported workspace size is backward-compatible with
		//        .    previous LAPACK versions. ====
		work.Set(0, complex(math.Max(work.GetRe(0), float64(max(1, *n))), rzero))
		return

	} else {
		//        ==== copy eigenvalues isolated by ZGEBAL ====
		if (*ilo) > 1 {
			goblas.Zcopy((*ilo)-1, h.CVector(0, 0, (*ldh)+1), w.Off(0, 1))
		}
		if (*ihi) < (*n) {
			goblas.Zcopy((*n)-(*ihi), h.CVector((*ihi), (*ihi), (*ldh)+1), w.Off((*ihi), 1))
		}

		//        ==== Initialize Z, if requested ====
		if initz {
			Zlaset('A', n, n, &zero, &one, z, ldz)
		}

		//        ==== Quick return if possible ====
		if (*ilo) == (*ihi) {
			w.Set((*ilo)-1, h.Get((*ilo)-1, (*ilo)-1))
			return
		}

		//        ==== ZLAHQR/ZLAQR0 crossover point ====
		nmin = Ilaenv(func() *int { y := 12; return &y }(), []byte("ZHSEQR"), []byte{job, compz}, n, ilo, ihi, lwork)
		nmin = max(ntiny, nmin)

		//        ==== ZLAQR0 for big matrices; ZLAHQR for small ones ====
		if (*n) > nmin {
			Zlaqr0(wantt, wantz, n, ilo, ihi, h, ldh, w, ilo, ihi, z, ldz, work, lwork, info)
		} else {
			//           ==== Small matrix ====
			Zlahqr(wantt, wantz, n, ilo, ihi, h, ldh, w, ilo, ihi, z, ldz, info)

			if (*info) > 0 {
				//              ==== A rare ZLAHQR failure!  ZLAQR0 sometimes succeeds
				//              .    when ZLAHQR fails. ====
				kbot = (*info)

				if (*n) >= nl {
					//                 ==== Larger matrices have enough subdiagonal scratch
					//                 .    space to call ZLAQR0 directly. ====
					Zlaqr0(wantt, wantz, n, ilo, &kbot, h, ldh, w, ilo, ihi, z, ldz, work, lwork, info)

				} else {
					//                 ==== Tiny matrices don't have enough subdiagonal
					//                 .    scratch space to benefit from ZLAQR0.  Hence,
					//                 .    tiny matrices must be copied into a larger
					//                 .    array before calling ZLAQR0. ====
					Zlacpy('A', n, n, h, ldh, hl, &nl)
					hl.Set((*n), (*n)-1, zero)
					Zlaset('A', &nl, toPtr(nl-(*n)), &zero, &zero, hl.Off(0, (*n)), &nl)
					Zlaqr0(wantt, wantz, &nl, ilo, &kbot, hl, &nl, w, ilo, ihi, z, ldz, workl, &nl, info)
					if wantt || (*info) != 0 {
						Zlacpy('A', n, n, hl, &nl, h, ldh)
					}
				}
			}
		}

		//        ==== Clear out the trash, if necessary. ====
		if (wantt || (*info) != 0) && (*n) > 2 {
			Zlaset('L', toPtr((*n)-2), toPtr((*n)-2), &zero, &zero, h.Off(2, 0), ldh)
		}

		//        ==== Ensure reported workspace size is backward-compatible with
		//        .    previous LAPACK versions. ====
		work.Set(0, complex(math.Max(float64(max(1, *n)), work.GetRe(0)), rzero))
	}
}
