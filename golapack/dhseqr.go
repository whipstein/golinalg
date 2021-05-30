package golapack

import (
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Dhseqr computes the eigenvalues of a Hessenberg matrix H
//    and, optionally, the matrices T and Z from the Schur decomposition
//    H = Z T Z**T, where T is an upper quasi-triangular matrix (the
//    Schur form), and Z is the orthogonal matrix of Schur vectors.
//
//    Optionally Z may be postmultiplied into an input orthogonal
//    matrix Q so that this routine can give the Schur factorization
//    of a matrix A which has been reduced to the Hessenberg form H
//    by the orthogonal matrix Q:  A = Q*H*Q**T = (QZ)*T*(QZ)**T.
func Dhseqr(job, compz byte, n, ilo, ihi *int, h *mat.Matrix, ldh *int, wr, wi *mat.Vector, z *mat.Matrix, ldz *int, work *mat.Vector, lwork, info *int) {
	var initz, lquery, wantt, wantz bool
	var one, zero float64
	var i, kbot, nl, nmin, ntiny int

	ntiny = 11

	nl = 49
	zero = 0.0
	one = 1.0
	workl := vf(nl)
	hl := mf(nl, nl, opts)

	//     ==== Decode and check the input parameters. ====
	wantt = job == 'S'
	initz = compz == 'I'
	wantz = initz || compz == 'V'
	work.Set(0, float64(maxint(1, *n)))
	lquery = (*lwork) == -1

	(*info) = 0
	if job != 'E' && !wantt {
		(*info) = -1
	} else if compz != 'N' && !wantz {
		(*info) = -2
	} else if (*n) < 0 {
		(*info) = -3
	} else if (*ilo) < 1 || (*ilo) > maxint(1, *n) {
		(*info) = -4
	} else if (*ihi) < minint(*ilo, *n) || (*ihi) > (*n) {
		(*info) = -5
	} else if (*ldh) < maxint(1, *n) {
		(*info) = -7
	} else if (*ldz) < 1 || (wantz && (*ldz) < maxint(1, *n)) {
		(*info) = -11
	} else if (*lwork) < maxint(1, *n) && !lquery {
		(*info) = -13
	}

	if (*info) != 0 {
		//        ==== Quick return in case of invalid argument. ====
		gltest.Xerbla([]byte("DHSEQR"), -(*info))
		return

	} else if (*n) == 0 {
		//        ==== Quick return in case N = 0; nothing to do. ====
		return

	} else if lquery {
		//        ==== Quick return in case of a workspace query ====
		Dlaqr0(wantt, wantz, n, ilo, ihi, h, ldh, wr, wi, ilo, ihi, z, ldz, work, lwork, info)
		//        ==== Ensure reported workspace size is backward-compatible with
		//        .    previous LAPACK versions. ====
		work.Set(0, maxf64(float64(maxint(1, *n)), work.Get(0)))
		return

	} else {
		//        ==== copy eigenvalues isolated by DGEBAL ====
		for i = 1; i <= (*ilo)-1; i++ {
			wr.Set(i-1, h.Get(i-1, i-1))
			wi.Set(i-1, zero)
		}
		for i = (*ihi) + 1; i <= (*n); i++ {
			wr.Set(i-1, h.Get(i-1, i-1))
			wi.Set(i-1, zero)
		}

		//        ==== Initialize Z, if requested ====
		if initz {
			Dlaset('A', n, n, &zero, &one, z, ldz)
		}

		//        ==== Quick return if possible ====
		if (*ilo) == (*ihi) {
			wr.Set((*ilo)-1, h.Get((*ilo)-1, (*ilo)-1))
			wi.Set((*ilo)-1, zero)
			return
		}

		//        ==== DLAHQR/DLAQR0 crossover point ====
		nmin = Ilaenv(func() *int { y := 12; return &y }(), []byte("DHSEQR"), []byte{job, compz}, n, ilo, ihi, lwork)
		nmin = maxint(ntiny, nmin)

		//        ==== DLAQR0 for big matrices; DLAHQR for small ones ====
		if (*n) > nmin {
			Dlaqr0(wantt, wantz, n, ilo, ihi, h, ldh, wr, wi, ilo, ihi, z, ldz, work, lwork, info)
		} else {
			//           ==== Small matrix ====
			Dlahqr(wantt, wantz, n, ilo, ihi, h, ldh, wr, wi, ilo, ihi, z, ldz, info)

			if (*info) > 0 {
				//              ==== A rare DLAHQR failure!  DLAQR0 sometimes succeeds
				//              .    when DLAHQR fails. ====
				kbot = (*info)

				if (*n) >= nl {
					//                 ==== Larger matrices have enough subdiagonal scratch
					//                 .    space to call DLAQR0 directly. ====
					Dlaqr0(wantt, wantz, n, ilo, &kbot, h, ldh, wr, wi, ilo, ihi, z, ldz, work, lwork, info)

				} else {
					//                 ==== Tiny matrices don't have enough subdiagonal
					//                 .    scratch space to benefit from DLAQR0.  Hence,
					//                 .    tiny matrices must be copied into a larger
					//                 .    array before calling DLAQR0. ====
					Dlacpy('A', n, n, h, ldh, hl, &nl)
					hl.Set((*n)+1-1, (*n)-1, zero)
					Dlaset('A', &nl, toPtr(nl-(*n)), &zero, &zero, hl.Off(0, (*n)+1-1), &nl)
					Dlaqr0(wantt, wantz, &nl, ilo, &kbot, hl, &nl, wr, wi, ilo, ihi, z, ldz, workl, &nl, info)
					if wantt || (*info) != 0 {
						Dlacpy('A', n, n, hl, &nl, h, ldh)
					}
				}
			}
		}

		//        ==== Clear out the trash, if necessary. ====
		if (wantt || (*info) != 0) && (*n) > 2 {
			Dlaset('L', toPtr((*n)-2), toPtr((*n)-2), &zero, &zero, h.Off(2, 0), ldh)
		}

		//        ==== Ensure reported workspace size is backward-compatible with
		//        .    previous LAPACK versions. ====
		work.Set(0, maxf64(float64(maxint(1, *n)), work.Get(0)))
	}
}
