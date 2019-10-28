# requires gawk

BEGIN {
    OFS="\t"; tok=1; sent=1;
    print "sentence_id", "token_id", "token", "surprisal";
}

match($0, /^\tp\( ([^[:space:]]+)/, ary ) {
    match($0, /([0-9]+\.[0-9]+(e-[0-9]+)?)/, ary2);
    print sent, tok, ary[1], -log(ary2[1])/log(2);
    tok += 1
}

/^$/ {
    sent += 1; tok = 1
}

