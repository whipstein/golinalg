package matgen

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Dlarot applies a (Givens) rotation to two adjacent rows or
//    columns, where one element of the first and/or last column/row
//    for use on matrices stored in some format other than GE, so
//    that elements of the matrix may be used or modified for which
//    no array element is provided.
//
//    One example is a symmetric matrix in SB format (bandwidth=4), for
//    which UPLO='L':  Two adjacent rows will have the format:
//
//    row j:     C> C> C> C> C> .  .  .  .
//    row j+1:      C> C> C> C> C> .  .  .  .
//
//    '*' indicates elements for which storage is provided,
//    '.' indicates elements for which no storage is provided, but
//    are not necessarily zero; their values are determined by
//    symmetry.  ' ' indicates elements which are necessarily zero,
//     and have no storage provided.
//
//    Those columns which have two '*'s can be handled by DROT.
//    Those columns which have no '*'s can be ignored, since as long
//    as the Givens rotations are carefully applied to preserve
//    symmetry, their values are determined.
//    Those columns which have one '*' have to be handled separately,
//    by using separate variables "p" and "q":
//
//    row j:     C> C> C> C> C> p  .  .  .
//    row j+1:   q  C> C> C> C> C> .  .  .  .
//
//    The element p would have to be set correctly, then that column
//    is rotated, setting p to its new value.  The next call to
//    DLAROT would rotate columns j and j+1, using p, and restore
//    symmetry.  The element q would start out being zero, and be
//    made non-zero by the rotation.  Later, rotations would presumably
//    be chosen to zero q out.
//
//    Typical Calling Sequences: rotating the i-th and (i+1)-st rows.
//    ------- ------- ---------
//
//      General dense matrix:
//
//              CALL DLAROT(.TRUE.,.FALSE.,.FALSE., N, C,S,
//                      A(i,1),LDA, DUMMY, DUMMY)
//
//      General banded matrix in GB format:
//
//              j = MAX(1, i-KL )
//              NL = MIN( N, i+KU+1 ) + 1-j
//              CALL DLAROT( .TRUE., i-KL.GE.1, i+KU.LT.N, NL, C,S,
//                      A(KU+i+1-j,j),LDA-1, XLEFT, XRIGHT )
//
//              [ note that i+1-j is just MIN(i,KL+1) ]
//
//      Symmetric banded matrix in SY format, bandwidth K,
//      lower triangle only:
//
//              j = MAX(1, i-K )
//              NL = MIN( K+1, i ) + 1
//              CALL DLAROT( .TRUE., i-K.GE.1, .TRUE., NL, C,S,
//                      A(i,j), LDA, XLEFT, XRIGHT )
//
//      Same, but upper triangle only:
//
//              NL = MIN( K+1, N-i ) + 1
//              CALL DLAROT( .TRUE., .TRUE., i+K.LT.N, NL, C,S,
//                      A(i,i), LDA, XLEFT, XRIGHT )
//
//      Symmetric banded matrix in SB format, bandwidth K,
//      lower triangle only:
//
//              [ same as for SY, except:]
//                  . . . .
//                      A(i+1-j,j), LDA-1, XLEFT, XRIGHT )
//
//              [ note that i+1-j is just MIN(i,K+1) ]
//
//      Same, but upper triangle only:
//                   . . .
//                      A(K+1,i), LDA-1, XLEFT, XRIGHT )
//
//      Rotating columns is just the transpose of rotating rows, except
//      for GB and SB: (rotating columns i and i+1)
//
//      GB:
//              j = MAX(1, i-KU )
//              NL = MIN( N, i+KL+1 ) + 1-j
//              CALL DLAROT( .TRUE., i-KU.GE.1, i+KL.LT.N, NL, C,S,
//                      A(KU+j+1-i,i),LDA-1, XTOP, XBOTTM )
//
//              [note that KU+j+1-i is just MAX(1,KU+2-i)]
//
//      SB: (upper triangle)
//
//                   . . . . . .
//                      A(K+j+1-i,i),LDA-1, XTOP, XBOTTM )
//
//      SB: (lower triangle)
//
//                   . . . . . .
//                      A(1,i),LDA-1, XTOP, XBOTTM )
func Dlarot(lrows, lleft, lright bool, nl *int, c, s *float64, a *mat.Vector, lda *int, xleft, xright *float64) {
	var iinc, inext, ix, iy, iyt, nt int

	xt := vf(2)
	yt := vf(2)

	//     Set up indices, arrays for ends
	if lrows {
		iinc = (*lda)
		inext = 1
	} else {
		iinc = 1
		inext = (*lda)
	}
	//
	if lleft {
		nt = 1
		ix = 1 + iinc
		iy = 2 + (*lda)
		xt.Set(0, a.Get(0))
		yt.Set(0, (*xleft))
	} else {
		nt = 0
		ix = 1
		iy = 1 + inext
	}
	//
	if lright {
		iyt = 1 + inext + ((*nl)-1)*iinc
		nt = nt + 1
		xt.Set(nt-1, (*xright))
		yt.Set(nt-1, a.Get(iyt-1))
	}

	//     Check for errors
	if (*nl) < nt {
		gltest.Xerbla([]byte("DLAROT"), 4)
		return
	}
	if (*lda) <= 0 || (!(lrows) && (*lda) < (*nl)-nt) {
		gltest.Xerbla([]byte("DLAROT"), 8)
		return
	}

	//     Rotate
	goblas.Drot(toPtr((*nl)-nt), a.Off(ix-1), &iinc, a.Off(iy-1), &iinc, c, s)
	goblas.Drot(&nt, xt, toPtr(1), yt, toPtr(1), c, s)

	//     Stuff values back into XLEFT, XRIGHT, etc.
	if lleft {
		a.Set(0, xt.Get(0))
		(*xleft) = yt.Get(0)
	}

	if lright {
		(*xright) = xt.Get(nt - 1)
		a.Set(iyt-1, yt.Get(nt-1))
	}
}
